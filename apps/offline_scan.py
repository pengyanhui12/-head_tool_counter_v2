"""离线扫描 Pipeline——完整流程编排"""
import argparse
import json
import sys
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.types import (
    Frame, KeyframeDecision, KeyframeTriggerContext,
    ConfirmationStatus, ReviewFlag,
)
from core.video_reader import VideoReader
from core.quality_evaluator import InitialKeyframeFallback, QualityEvaluator
from core.frame_buffer import FrameBuffer
from core.feature_matcher import FeatureMatcher
from core.homography_graph import HomographyGraph
from core.keyframe_selector import KeyframeSelector
from core.detector import Detector
from core.detection_fusion import DetectionFusion
from core.simple_tracker import SimpleDetectionTracker
from core.global_projector import GlobalProjector
from core.object_associator import ObjectAssociator
from core.coverage_map import CoverageMap
from core.status_panel import StatusPanel
from core.report_generator import ReportGenerator, get_display_objects
from core.evidence_extractor import EvidenceExtractor
from core.session_store import SessionStore
from core.config_loader import ConfigLoader
from core.recovery_manager import RecoveryManager, RecoveryState
from core.debug_events import DebugStats, PerfTimer, TimedMatcher


PERFORMANCE_LABELS = {
    "initialization_ms": "Initialization",
    "video_decode_ms": "Video decode",
    "quality_ms": "Quality evaluation",
    "l2_inference_ms": "L2 inference",
    "tracker_preview_ms": "Tracker preview",
    "tracker_update_ms": "Tracker update",
    "keyframe_decision_ms": "Keyframe decision",
    "feature_match_ms": "Feature matching",
    "recovery_ms": "Recovery",
    "l1_inference_ms": "L1 inference",
    "l3_inference_ms": "L3 inference",
    "fusion_ms": "Fusion",
    "graph_update_ms": "Graph update",
    "projection_ms": "Projection",
    "association_ms": "Association",
    "coverage_update_ms": "Coverage update",
    "end_window_ms": "End-window processing",
    "final_review_ms": "Final review",
    "mosaic_ms": "Global mosaic",
    "evidence_ms": "Evidence extraction",
    "session_store_ms": "Session storage",
    "report_ms": "Report generation",
    "event_log_io_ms": "Event/log I/O",
    "debug_image_write_ms": "Debug image I/O",
}


def resolve_performance_enabled(config: dict, override: bool | None) -> bool:
    if override is not None:
        return override
    return bool(config.get("enable_performance_stats", False))


def build_report_snapshot(report_generator, objects) -> tuple[dict, str]:
    """Create JSON and CSV output from one JSON report snapshot."""
    return (
        report_generator.generate_json_report(),
        report_generator.generate_csv_report(objects),
    )


def print_performance_stats(
    stats: DebugStats,
    wall_ms: float,
    total_frames: int,
) -> None:
    total_fps = total_frames / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
    print("\nPerformance breakdown:")
    print(f"  End-to-end: {wall_ms / 1000.0:.3f}s ({total_fps:.1f} FPS)")
    print(f"  {'Stage':<24} {'Total(ms)':>10} {'Calls':>8} {'Avg(ms)':>10} {'Wall%':>8}")
    for field, total, calls, average in stats.performance_rows():
        wall_pct = total / wall_ms * 100.0 if wall_ms > 0 else 0.0
        print(
            f"  {PERFORMANCE_LABELS[field]:<24} {total:>10.1f} "
            f"{calls:>8d} {average:>10.3f} {wall_pct:>7.1f}%"
        )
    unaccounted = stats.unaccounted_ms(wall_ms)
    print(f"  {'Unaccounted overhead':<24} {unaccounted:>10.1f} "
          f"{'-':>8} {'-':>10} {unaccounted / wall_ms * 100.0:>7.1f}%")
    print(f"  Accounted coverage: {stats.coverage_ratio(wall_ms) * 100.0:.1f}%")


def build_raw_detections(tracked_detections, keyframe_id: int,
                          sharpness: float, mapping_quality: float):
    from core.types import RawDetection
    raw = []
    for td in tracked_detections:
        c = td.candidate
        x1, y1, x2, y2 = c.bbox
        raw.append(RawDetection(
            frame_id=c.frame_id,
            keyframe_id=keyframe_id,
            track_id=td.track_id,
            bbox=c.bbox,
            center=((x1 + x2) / 2, (y1 + y2) / 2),
            corners=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            class_id=c.class_id,
            class_name=c.class_name,
            confidence=c.confidence,
            sharpness=sharpness,
            mapping_quality=mapping_quality,
            source=c.source,
        ))
    return raw


def build_untracked_raw_detections(candidates, keyframe_id: int,
                                   sharpness: float, mapping_quality: float):
    from core.types import RawDetection
    raw = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.bbox
        raw.append(RawDetection(
            frame_id=candidate.frame_id, keyframe_id=keyframe_id,
            track_id=None, bbox=candidate.bbox,
            center=((x1 + x2) / 2, (y1 + y2) / 2),
            corners=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            class_id=candidate.class_id, class_name=candidate.class_name,
            confidence=candidate.confidence, sharpness=sharpness,
            mapping_quality=mapping_quality, source=candidate.source,
        ))
    return raw


def should_run_l2(fc: int, interval: int = 3) -> bool:
    return fc % interval == 0


def _build_associator(acfg: dict) -> ObjectAssociator:
    return ObjectAssociator(
        max_position_distance_px=acfg.get("max_position_distance_px", 120.0),
        position_weight=acfg.get("position_weight", 0.55),
        overlap_weight=acfg.get("overlap_weight", 0.20),
        size_weight=acfg.get("size_weight", 0.10),
        class_weight=acfg.get("class_weight", 0.15),
        max_cost=acfg.get("max_cost", 0.75),
        min_observations_confirmed=acfg.get("min_observations_confirmed", 3),
        min_keyframes_confirmed=acfg.get("min_keyframes_confirmed", 2),
        min_top_class_ratio=acfg.get("min_top_class_ratio", 0.60),
        max_votes_per_track=acfg.get("max_votes_per_track", 3),
        class_compatibility=acfg.get("class_compatibility", {}),
        online_gate_ratio=acfg.get("online_gate_ratio", 0.5),
        per_class_gate_ratios=acfg.get("per_class_gate_ratios", {}),
        per_class_position_gates=acfg.get("per_class_position_gates", {}),
        track_reactivate_max_gap_frames=acfg.get(
            "track_reactivate_max_gap_frames", 15
        ),
        centroid_distance_threshold=acfg.get(
            "centroid_distance_threshold", 30.0
        ),
        independent_co_occurrence_max_containment=acfg.get(
            "independent_co_occurrence_max_containment", 0.25
        ),
        partial_duplicate_min_containment=acfg.get(
            "partial_duplicate_min_containment", 0.75
        ),
        partial_duplicate_max_normalized_distance=acfg.get(
            "partial_duplicate_max_normalized_distance", 0.75
        ),
        partial_duplicate_max_absolute_distance_px=acfg.get(
            "partial_duplicate_max_absolute_distance_px", 80.0
        ),
        partial_duplicate_min_mapping_quality=acfg.get(
            "partial_duplicate_min_mapping_quality", 0.50
        ),
        partial_duplicate_max_area_ratio=acfg.get(
            "partial_duplicate_max_area_ratio", 0.60
        ),
        partial_duplicate_min_candidate_margin=acfg.get(
            "partial_duplicate_min_candidate_margin", 0.15
        ),
        debug_mode=False,
    )


def run_pipeline(
    video_path: str,
    config_dir: str,
    output_dir: str,
    performance: bool | None = None,
):
    _proj_root = Path(__file__).resolve().parent.parent
    cfg_path = (Path(config_dir) if Path(config_dir).is_absolute()
                else _proj_root / config_dir)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config directory not found: {cfg_path}")

    # 使用统一的 ConfigLoader
    loader = ConfigLoader(cfg_path)
    cfg = loader.pipeline
    mcfg = loader.matcher
    acfg = loader.associator
    tcfg = loader.tracker
    detector_cfg = loader.detector
    fcfg = detector_cfg.get("fusion", {})
    l3cfg = detector_cfg.get("l3", {})
    performance_enabled = resolve_performance_enabled(cfg, performance)
    stats = DebugStats() if performance_enabled else None
    wall_start = time.perf_counter()
    initialization_start = time.perf_counter()

    def timer(field_name: str):
        return PerfTimer(stats, field_name) if stats is not None else nullcontext()

    l2_interval = cfg.get("l2_interval_frames", 3)
    max_interval = cfg.get("max_keyframe_interval_frames", 30)
    end_window = cfg.get("end_window_frames", 30)
    min_kf_interval = cfg.get("min_keyframe_interval_frames", 5)

    # Modules
    reader = VideoReader(video_path, max_fps=cfg.get("max_input_fps", 30))
    quality = QualityEvaluator(
        sharpness_threshold=cfg.get("sharpness_threshold", 20.0),
        detection_sharpness_threshold=cfg.get(
            "detection_sharpness_threshold", 63.0
        ),
        quality_evaluation_scale=cfg.get("quality_evaluation_scale", 0.5),
        dark_pixel_threshold=cfg.get("dark_pixel_threshold", 10),
        bright_pixel_threshold=cfg.get("bright_pixel_threshold", 245),
    )
    frame_buffer = FrameBuffer(max_size=end_window)
    matcher = FeatureMatcher(
        min_good_matches=mcfg.get("min_good_matches", 20),
        min_inliers=mcfg.get("min_inliers", 30),
        min_inlier_ratio=mcfg.get("min_inlier_ratio", 0.30),
        max_reprojection_error=mcfg.get("max_reprojection_error_px", 3.0),
        min_occupied_quadrants=mcfg.get("min_occupied_quadrants", 3),
        min_inlier_bbox_area_ratio=mcfg.get("min_inlier_bbox_area_ratio", 0.15),
        roi_center_ratio=mcfg.get("roi_center_ratio", 0.70),
        max_projected_area_ratio=mcfg.get("max_projected_area_ratio", 50.0),
        min_projected_area_ratio=mcfg.get("min_projected_area_ratio", 0.01),
    )
    if stats is not None:
        matcher = TimedMatcher(matcher, stats)
    graph = HomographyGraph()
    selector = KeyframeSelector(
        max_interval=max_interval,
        end_window_frames=end_window,
        matcher=matcher,
        min_keyframe_interval_frames=min_kf_interval,
        emergency_keyframe_interval_frames=cfg.get("emergency_keyframe_interval_frames", 2),
    )
    detector = Detector(model_path=str(_proj_root / "models" / "best.pt"))
    fusion = DetectionFusion(
        iou_threshold=fcfg.get("iou_threshold", 0.65),
        center_merge_distance_px=fcfg.get("center_merge_distance_px", 40.0),
        center_merge_min_ios=fcfg.get("center_merge_min_ios", 0.30),
        per_class_center_merge_distances=fcfg.get(
            "per_class_center_merge_distances", {}
        ),
    )
    tracker = SimpleDetectionTracker(
        max_missed_detection_frames=tcfg.get("max_missed_detection_frames", 5),
        lost_reactivation_frames=tcfg.get("lost_reactivation_frames", 10),
        min_iou=tcfg.get("min_iou", 0.20),
        max_center_distance_ratio=tcfg.get("max_center_distance_ratio", 0.20),
        iou_weight=tcfg.get("iou_weight", 0.60),
        center_weight=tcfg.get("center_weight", 0.40),
        class_compatibility=acfg.get("class_compatibility", {}),
        inactive_min_iou=tcfg.get("inactive_min_iou", 0.30),
        inactive_max_center_distance_ratio=tcfg.get("inactive_max_center_distance_ratio", 0.12),
        quality_drop_trigger_ratio=tcfg.get("quality_drop_trigger_ratio", 0.70),
        quality_drop_rearm_ratio=tcfg.get("quality_drop_rearm_ratio", 0.85),
        quality_drop_min_history=tcfg.get("quality_drop_min_history", 5),
        new_detection_confirmation_runs=tcfg.get(
            "new_detection_confirmation_runs", 3
        ),
    )
    projector = GlobalProjector()
    associator = _build_associator(acfg)
    coverage = CoverageMap(grid_resolution=100)
    status = StatusPanel()
    recovery_mgr = RecoveryManager(matcher=matcher)
    if stats is not None:
        stats.add_timing(
            "initialization_ms",
            (time.perf_counter() - initialization_start) * 1000.0,
        )

    # State
    last_keyframe: Frame | None = None
    last_keyframe_frame_id: int = -1
    last_keyframe_node_id: int | None = None
    end_window_deque: deque[Frame] = deque(maxlen=end_window)
    processed_keyframe_frame_ids: set[int] = set()
    fc = 0
    initial_fallback = InitialKeyframeFallback(
        min_sharpness=cfg.get("detection_sharpness_threshold", 63.0),
        max_interval_frames=max_interval,
    )

    print(f"Processing: {video_path}")
    frame_iterator = iter(reader.read())
    while True:
        try:
            with timer("video_decode_ms"):
                frame = next(frame_iterator)
        except StopIteration:
            break
        fc += 1
        with timer("quality_ms"):
            frame = quality.evaluate(frame)
        mapping_eligible = quality.is_acceptable(frame)
        historical_keyframe = False
        if last_keyframe is None:
            initial_fallback.consider(frame)
            fallback_frame = initial_fallback.select(frame.frame_id)
            if not mapping_eligible and fallback_frame is not None:
                historical_keyframe = fallback_frame.frame_id < tracker.current_frame_id
                frame = fallback_frame
                mapping_eligible = True
            if mapping_eligible:
                initial_fallback.clear()

        if not quality.is_acceptable_for_detection(frame):
            tracker.advance_frame(frame.frame_id)
            recovery_mgr.cache_frame(frame)
            continue

        frame_buffer.push(frame)
        end_window_deque.append(frame)

        # L2 detection
        l2_was_run = should_run_l2(fc, l2_interval)
        l2_candidates = []
        if l2_was_run:
            with timer("l2_inference_ms"):
                l2_candidates = detector.detect(
                    image=frame.image, level="L2", frame_id=frame.frame_id,
                )

        with timer("tracker_preview_ms"):
            preview = tracker.preview(l2_candidates, l2_was_run=l2_was_run)

        if not mapping_eligible:
            if l2_was_run:
                with timer("tracker_update_ms"):
                    tracker.update(l2_candidates, frame_id=frame.frame_id)
            recovery_mgr.cache_frame(frame)
            continue

        l3_regions = Detector.select_l3_regions(l2_candidates, l3cfg)
        trigger_context = KeyframeTriggerContext(
            max_interval_reached=(frame.frame_id - last_keyframe_frame_id >= max_interval),
            l2_new_unmatched_detection=preview.l2_new_unmatched_detection,
            track_quality_drop=preview.track_quality_drop,
            l3_required=bool(l3_regions),
            l3_regions=l3_regions,
        )

        match_before = stats.feature_match_ms if stats is not None else 0.0
        decision_start = time.perf_counter()
        keyframe_result = selector.evaluate(
            frame=frame,
            previous_keyframe=last_keyframe,
            trigger_context=trigger_context,
        )
        if stats is not None:
            elapsed = (time.perf_counter() - decision_start) * 1000.0
            stats.add_timing(
                "keyframe_decision_ms",
                max(0.0, elapsed - (stats.feature_match_ms - match_before)),
            )

        if keyframe_result.decision == KeyframeDecision.ACCEPTED:
            with timer("graph_update_ms"):
                if last_keyframe is None:
                    keyframe_id = graph.add_first_keyframe(frame_id=frame.frame_id)
                    frame.mapping_quality = 1.0
                else:
                    keyframe_id = graph.add_keyframe(
                        frame_id=frame.frame_id,
                        H_current_to_parent=keyframe_result.H_current_to_previous,
                        parent_node_id=last_keyframe_node_id,
                    )
                    if keyframe_result.match_result is not None:
                        frame.mapping_quality = float(keyframe_result.match_result.inlier_ratio)
                H_kf_to_global = graph.get_transform(keyframe_id)

            with timer("l1_inference_ms"):
                l1_candidates = detector.detect(
                    image=frame.image, level="L1", frame_id=frame.frame_id,
                )

            l3_candidates = []
            if trigger_context.l3_required:
                with timer("l3_inference_ms"):
                    l3_candidates = detector.detect(
                        image=frame.image, level="L3", frame_id=frame.frame_id,
                        regions=trigger_context.l3_regions,
                    )

            with timer("fusion_ms"):
                fused_candidates = fusion.fuse(l1=l1_candidates, l3=l3_candidates)
            tracked = []
            if not historical_keyframe:
                with timer("tracker_update_ms"):
                    tracked = tracker.update(fused_candidates, frame_id=frame.frame_id)

            if historical_keyframe:
                raw_detections = build_untracked_raw_detections(
                    fused_candidates, keyframe_id, frame.sharpness_score,
                    frame.mapping_quality,
                )
            else:
                raw_detections = build_raw_detections(
                    tracked_detections=tracked,
                    keyframe_id=keyframe_id,
                    sharpness=frame.sharpness_score,
                    mapping_quality=frame.mapping_quality,
                )

            with timer("projection_ms"):
                global_detections = [
                    projector.project(
                        detection=rd,
                        H_keyframe_to_global=H_kf_to_global,
                        transform_version=graph.transform_version,
                    )
                    for rd in raw_detections
                ]

            with timer("association_ms"):
                associator.ingest_frame(
                    frame_id=frame.frame_id,
                    global_detections=global_detections,
                )

            with timer("coverage_update_ms"):
                projected_fov = projector.project_frame_corners(
                    image_shape=frame.image.shape,
                    H_keyframe_to_global=H_kf_to_global,
                )
                coverage.update(frame.frame_id, projected_fov)

            processed_keyframe_frame_ids.add(frame.frame_id)
            recovery_mgr.reset()
            last_keyframe = frame
            last_keyframe_frame_id = frame.frame_id
            last_keyframe_node_id = keyframe_id

        elif keyframe_result.decision == KeyframeDecision.RECOVERY:
            recovery_match_before = stats.feature_match_ms if stats is not None else 0.0
            recovery_start = time.perf_counter()
            recovery_result = recovery_mgr.recover(
                current_frame=frame, previous_keyframe=last_keyframe,
                frame_buffer=frame_buffer, graph=graph,
                keyframe_images=None,
            )
            if stats is not None:
                recovery_elapsed = (time.perf_counter() - recovery_start) * 1000.0
                stats.add_timing(
                    "recovery_ms",
                    max(
                        0.0,
                        recovery_elapsed
                        - (stats.feature_match_ms - recovery_match_before),
                    ),
                )

            if recovery_result.state == RecoveryState.RECOVERED and recovery_result.H_current_to_anchor is not None:
                parent_id = recovery_result.anchor_node_id or last_keyframe_node_id
                keyframe_id = graph.add_keyframe(
                    frame_id=frame.frame_id,
                    H_current_to_parent=recovery_result.H_current_to_anchor,
                    parent_node_id=parent_id,
                )
                frame.mapping_quality = 0.5
                H_kf_to_global = graph.get_transform(keyframe_id)

                l1_candidates = detector.detect(
                    image=frame.image, level="L1", frame_id=frame.frame_id,
                )
                fused_candidates = fusion.fuse(l1=l1_candidates, l3=[])
                tracked = tracker.update(fused_candidates, frame_id=frame.frame_id)

                raw_detections = build_raw_detections(
                    tracked_detections=tracked,
                    keyframe_id=keyframe_id,
                    sharpness=frame.sharpness_score,
                    mapping_quality=frame.mapping_quality,
                )
                global_detections = [
                    projector.project(
                        detection=rd,
                        H_keyframe_to_global=H_kf_to_global,
                        transform_version=graph.transform_version,
                    )
                    for rd in raw_detections
                ]
                associator.ingest_frame(
                    frame_id=frame.frame_id,
                    global_detections=global_detections,
                )
                processed_keyframe_frame_ids.add(frame.frame_id)
                recovery_mgr.reset()
                last_keyframe = frame
                last_keyframe_frame_id = frame.frame_id
                last_keyframe_node_id = keyframe_id
            else:
                recovery_mgr.cache_frame(frame)
                recovery_mgr.cache_detections(frame.frame_id, l2_candidates, frame.sharpness_score)

        elif l2_was_run:
            with timer("tracker_update_ms"):
                tracker.update(l2_candidates, frame_id=frame.frame_id)

    # Post-video
    print(f"\nProcessed {fc} frames, {graph.num_keyframes} keyframes accepted.")

    # End window
    end_match_before = stats.feature_match_ms if stats is not None else 0.0
    end_start = time.perf_counter()
    end_kfs = selector.select_end_keyframes(list(end_window_deque))
    if stats is not None:
        stats.add_timing(
            "end_window_ms",
            max(
                0.0,
                (time.perf_counter() - end_start) * 1000.0
                - (stats.feature_match_ms - end_match_before),
            ),
        )
    for ekf in end_kfs:
        if ekf.frame_id in processed_keyframe_frame_ids:
            continue
        end_eval_match_before = stats.feature_match_ms if stats is not None else 0.0
        end_eval_start = time.perf_counter()
        result = selector.evaluate(ekf, last_keyframe,
                                   KeyframeTriggerContext(force_end_candidate=True))
        if stats is not None:
            stats.add_timing(
                "end_window_ms",
                max(
                    0.0,
                    (time.perf_counter() - end_eval_start) * 1000.0
                    - (stats.feature_match_ms - end_eval_match_before),
                ),
            )
        if result.decision == KeyframeDecision.ACCEPTED:
            try:
                keyframe_id = graph.add_keyframe(
                    frame_id=ekf.frame_id,
                    H_current_to_parent=result.H_current_to_previous,
                    parent_node_id=last_keyframe_node_id,
                )
                H_kf_to_global = graph.get_transform(keyframe_id)
                l1_candidates = detector.detect(image=ekf.image, level="L1", frame_id=ekf.frame_id)
                fused_candidates = fusion.fuse(l1=l1_candidates, l3=[])
                raw_detections = build_untracked_raw_detections(
                    fused_candidates, keyframe_id, ekf.sharpness_score,
                    ekf.mapping_quality,
                )
                global_detections = [
                    projector.project(detection=rd, H_keyframe_to_global=H_kf_to_global,
                                      transform_version=graph.transform_version)
                    for rd in raw_detections
                ]
                associator.ingest_frame(frame_id=ekf.frame_id, global_detections=global_detections)
                processed_keyframe_frame_ids.add(ekf.frame_id)
                last_keyframe = ekf
                last_keyframe_frame_id = ekf.frame_id
                last_keyframe_node_id = keyframe_id
            except (ValueError, KeyError) as e:
                print(f"  End KF failed: frame {ekf.frame_id} — {e}")

    # Final review
    with timer("final_review_ms"):
        associator.final_review()
        associator.map.assign_persistent_ids()

    with timer("report_ms"):
        gen = ReportGenerator(object_map=associator.map)
        all_objects = associator.map.get_all()
        display_objects = get_display_objects(all_objects)
        gen.find_evidence_frames(display_objects)
        json_report, csv_report = build_report_snapshot(gen, all_objects)

    # Save graph
    node_data = []
    for node_id, frame_id, H in graph.nodes:
        node_data.append({
            "node_id": node_id,
            "frame_id": frame_id,
            "H_to_global": H.tolist(),
        })

    from core.global_mosaic import generate_global_mosaic
    with timer("mosaic_ms"):
        generate_global_mosaic(
            video_path=video_path,
            graph=graph,
            objects=display_objects,
            output_dir=output_dir,
        )

    extractor = EvidenceExtractor()
    with timer("evidence_ms"):
        extractor.extract(video_path, display_objects, output_dir)

    with timer("session_store_ms"):
        store = SessionStore(output_dir)
        store.create_session(video_path)
        store.save_report(json_report)

    out = Path(output_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    with timer("report_ms"):
        json_path = out / "reports" / "report.json"
        json_path.write_text(
            json.dumps(json_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        csv_path = out / "reports" / "report.csv"
        csv_path.write_text(csv_report, encoding="utf-8")
    print(f"\nDone.")
    print(f"  Counted objects: {json_report['total_objects']}")
    print(f"  Review candidates: {json_report['review_candidate_count']}")
    print(
        "  Likely partial duplicates: "
        f"{json_report['likely_partial_duplicate_count']}"
    )
    print(f"  CONFIRMED: {json_report['confirmed_count']}")
    print(f"  UNCERTAIN: {json_report['uncertain_count']}")
    print(f"  REJECTED: {json_report['rejected_count']}")
    print(f"  Tracker time regressions: {tracker.time_regressions}")
    print(f"Reports: {out}/reports/")
    if stats is not None:
        print_performance_stats(
            stats,
            wall_ms=(time.perf_counter() - wall_start) * 1000.0,
            total_frames=fc,
        )


def main():
    parser = argparse.ArgumentParser(description="Head Tool Counter - Offline Scan")
    parser.add_argument("--video", default=r"D:\杭州供电段\头戴设备作业工具识别\260814拍摄测试\test.mp4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", default="./outputs1")
    parser.add_argument(
        "--performance",
        action="store_true",
        help="输出各处理环节的耗时、调用次数和平均耗时",
    )
    args = parser.parse_args()
    run_pipeline(
        args.video,
        args.config_dir,
        args.output_dir,
        performance=True if args.performance else None,
    )


if __name__ == "__main__":
    main()
