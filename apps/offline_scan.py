"""离线扫描 Pipeline——完整流程编排"""
import argparse
import json
import sys
import time
from collections import deque
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
from core.performance_profiler import (
    PerformanceProfiler,
    resolve_performance_enabled,
)
from core.class_summary import (
    build_class_summary,
    format_class_summary,
    save_class_summary,
)


def build_report_snapshot(report_generator, objects) -> tuple[dict, str]:
    """Create JSON and CSV output from one JSON report snapshot."""
    return (
        report_generator.generate_json_report(),
        report_generator.generate_csv_report(objects),
    )


def _emit_class_summary_if_enabled(
    report: dict,
    output_dir: str | Path,
    enabled: bool,
) -> Path | None:
    """按开关输出并保存类别摘要；关闭时不产生任何副作用。"""
    if not enabled:
        return None

    summary = build_class_summary(report)
    output_path = save_class_summary(summary, output_dir)
    print("\n" + format_class_summary(summary))
    print(f"Saved: {output_path}")
    return output_path


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


def is_mapping_eligible(quality: QualityEvaluator, frame: Frame) -> bool:
    """使用清晰度和曝光的严格条件判断帧是否适合建图。"""
    return quality.is_acceptable_for_mapping(frame)


def _build_detector(detector_cfg: dict, project_root: Path) -> Detector:
    """根据 detector.yaml 完整装配检测器。"""
    model_cfg = detector_cfg.get("model", {})
    levels = detector_cfg.get("levels", {})
    l1_cfg = levels.get("L1", {})
    l2_cfg = levels.get("L2", {})
    l3_cfg = levels.get("L3", {})

    model_path = Path(model_cfg.get("path", "models/best.pt"))
    if not model_path.is_absolute():
        model_path = project_root / model_path

    return Detector(
        model_path=str(model_path),
        device=model_cfg.get("device", "cuda:0"),
        l1_imgsz=l1_cfg.get("imgsz", 1280),
        l1_conf=l1_cfg.get("conf", 0.15),
        l1_iou=l1_cfg.get("iou", 0.65),
        l2_imgsz=l2_cfg.get("imgsz", 640),
        l2_conf=l2_cfg.get("conf", 0.10),
        l2_iou=l2_cfg.get("iou", 0.65),
        l3_imgsz=l3_cfg.get("imgsz", 1280),
        l3_conf=l3_cfg.get("conf", 0.10),
        l3_iou=l3_cfg.get("iou", 0.65),
    )


def _build_coverage_map(coverage_cfg: dict) -> CoverageMap:
    """根据 coverage.yaml 装配覆盖地图。"""
    return CoverageMap(
        grid_resolution=coverage_cfg.get("grid_resolution", 100),
        minimum_valid_polygon_area=coverage_cfg.get(
            "minimum_valid_polygon_area", 0.0
        ),
        target_coverage_ratio=coverage_cfg.get("target_coverage_ratio", 0.95),
    )


def _build_matcher(matcher_cfg: dict) -> FeatureMatcher:
    """使用正式Matcher配置构造可缓存的SIFT匹配器。"""
    return FeatureMatcher(
        min_good_matches=matcher_cfg.get("min_good_matches", 20),
        min_inliers=matcher_cfg.get("min_inliers", 30),
        min_inlier_ratio=matcher_cfg.get("min_inlier_ratio", 0.30),
        max_reprojection_error=matcher_cfg.get(
            "max_reprojection_error_px", 3.0
        ),
        min_occupied_quadrants=matcher_cfg.get(
            "min_occupied_quadrants", 3
        ),
        min_inlier_bbox_area_ratio=matcher_cfg.get(
            "min_inlier_bbox_area_ratio", 0.15
        ),
        roi_center_ratio=matcher_cfg.get("roi_center_ratio", 0.70),
        max_projected_area_ratio=matcher_cfg.get(
            "max_projected_area_ratio", 50.0
        ),
        min_projected_area_ratio=matcher_cfg.get(
            "min_projected_area_ratio", 0.01
        ),
        feature_cache_size=matcher_cfg.get("feature_cache_size", 4),
    )


def _build_keyframe_selector(
    pipeline_cfg: dict,
    matcher,
) -> KeyframeSelector:
    """集中接入关键帧间隔和尾窗预筛配置。"""
    return KeyframeSelector(
        max_interval=pipeline_cfg.get(
            "max_keyframe_interval_frames", 30
        ),
        end_window_frames=pipeline_cfg.get("end_window_frames", 30),
        end_window_match_candidates=pipeline_cfg.get(
            "end_window_match_candidates", 6
        ),
        matcher=matcher,
        min_keyframe_interval_frames=pipeline_cfg.get(
            "min_keyframe_interval_frames", 5
        ),
        emergency_keyframe_interval_frames=pipeline_cfg.get(
            "emergency_keyframe_interval_frames", 2
        ),
    )


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
    class_summary: bool = False,
):
    # Pipeline 墙钟计时从入口开始，明确包含配置加载和模块初始化。
    pipeline_started_at = time.perf_counter()
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
    coverage_cfg = loader.coverage
    fcfg = detector_cfg.get("fusion", {})
    l3cfg = detector_cfg.get("l3", {})
    performance_enabled = resolve_performance_enabled(cfg, performance)
    profiler = PerformanceProfiler(
        enabled=performance_enabled,
        started_at=pipeline_started_at,
    )

    l2_interval = cfg.get("l2_interval_frames", 3)
    max_interval = cfg.get("max_keyframe_interval_frames", 30)
    end_window = cfg.get("end_window_frames", 30)

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
    matcher = _build_matcher(mcfg)
    matcher = profiler.wrap_matcher(matcher)
    graph = HomographyGraph()
    selector = _build_keyframe_selector(cfg, matcher)
    detector = _build_detector(detector_cfg, _proj_root)
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
    coverage = _build_coverage_map(coverage_cfg)
    status = StatusPanel()
    recovery_mgr = RecoveryManager(matcher=matcher)
    profiler.record(
        "initialization",
        (time.perf_counter() - pipeline_started_at) * 1000.0,
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
            frame = profiler.measure_success(
                "video_decode",
                lambda: next(frame_iterator),
            )
        except StopIteration:
            break
        fc += 1
        with profiler.measure("quality"):
            frame = quality.evaluate(frame)
        mapping_eligible = is_mapping_eligible(quality, frame)
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
            with profiler.measure("l2_inference"):
                l2_candidates = detector.detect(
                    image=frame.image, level="L2", frame_id=frame.frame_id,
                )

        with profiler.measure("tracker_preview"):
            preview = tracker.preview(l2_candidates, l2_was_run=l2_was_run)

        if not mapping_eligible:
            if l2_was_run:
                with profiler.measure("tracker_update"):
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

        match_before = profiler.stage_total("feature_match")
        decision_start = time.perf_counter()
        keyframe_result = selector.evaluate(
            frame=frame,
            previous_keyframe=last_keyframe,
            trigger_context=trigger_context,
        )
        profiler.record_exclusive(
            "keyframe_decision",
            started_at=decision_start,
            excluded_stage="feature_match",
            excluded_before_ms=match_before,
        )

        if keyframe_result.decision == KeyframeDecision.ACCEPTED:
            with profiler.measure("graph_update"):
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

            with profiler.measure("l1_inference"):
                l1_candidates = detector.detect(
                    image=frame.image, level="L1", frame_id=frame.frame_id,
                )

            l3_candidates = []
            if trigger_context.l3_required:
                with profiler.measure("l3_inference"):
                    l3_candidates = detector.detect(
                        image=frame.image, level="L3", frame_id=frame.frame_id,
                        regions=trigger_context.l3_regions,
                    )

            with profiler.measure("fusion"):
                fused_candidates = fusion.fuse(l1=l1_candidates, l3=l3_candidates)
            tracked = []
            if not historical_keyframe:
                with profiler.measure("tracker_update"):
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

            with profiler.measure("projection"):
                global_detections = [
                    projector.project(
                        detection=rd,
                        H_keyframe_to_global=H_kf_to_global,
                        transform_version=graph.transform_version,
                    )
                    for rd in raw_detections
                ]

            with profiler.measure("association"):
                associator.ingest_frame(
                    frame_id=frame.frame_id,
                    global_detections=global_detections,
                )

            with profiler.measure("coverage_update"):
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
            recovery_match_before = profiler.stage_total("feature_match")
            recovery_start = time.perf_counter()
            recovery_result = recovery_mgr.recover(
                current_frame=frame, previous_keyframe=last_keyframe,
                frame_buffer=frame_buffer, graph=graph,
                keyframe_images=None,
            )
            profiler.record_exclusive(
                "recovery",
                started_at=recovery_start,
                excluded_stage="feature_match",
                excluded_before_ms=recovery_match_before,
            )

            if recovery_result.state == RecoveryState.RECOVERED and recovery_result.H_current_to_anchor is not None:
                with profiler.measure("graph_update"):
                    parent_id = recovery_result.anchor_node_id or last_keyframe_node_id
                    keyframe_id = graph.add_keyframe(
                        frame_id=frame.frame_id,
                        H_current_to_parent=recovery_result.H_current_to_anchor,
                        parent_node_id=parent_id,
                    )
                    frame.mapping_quality = 0.5
                    H_kf_to_global = graph.get_transform(keyframe_id)

                with profiler.measure("l1_inference"):
                    l1_candidates = detector.detect(
                        image=frame.image, level="L1", frame_id=frame.frame_id,
                    )
                with profiler.measure("fusion"):
                    fused_candidates = fusion.fuse(l1=l1_candidates, l3=[])
                with profiler.measure("tracker_update"):
                    tracked = tracker.update(
                        fused_candidates,
                        frame_id=frame.frame_id,
                    )

                raw_detections = build_raw_detections(
                    tracked_detections=tracked,
                    keyframe_id=keyframe_id,
                    sharpness=frame.sharpness_score,
                    mapping_quality=frame.mapping_quality,
                )
                with profiler.measure("projection"):
                    global_detections = [
                        projector.project(
                            detection=rd,
                            H_keyframe_to_global=H_kf_to_global,
                            transform_version=graph.transform_version,
                        )
                        for rd in raw_detections
                    ]
                with profiler.measure("association"):
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
            with profiler.measure("tracker_update"):
                tracker.update(l2_candidates, frame_id=frame.frame_id)

    # Post-video
    print(f"\nProcessed {fc} frames, {graph.num_keyframes} keyframes accepted.")

    # End window
    end_match_before = profiler.stage_total("feature_match")
    end_start = time.perf_counter()
    end_kfs = selector.select_end_keyframes(list(end_window_deque))
    profiler.record_exclusive(
        "end_window",
        started_at=end_start,
        excluded_stage="feature_match",
        excluded_before_ms=end_match_before,
    )
    for ekf in end_kfs:
        if ekf.frame_id in processed_keyframe_frame_ids:
            continue
        end_eval_match_before = profiler.stage_total("feature_match")
        end_eval_start = time.perf_counter()
        result = selector.evaluate(ekf, last_keyframe,
                                   KeyframeTriggerContext(force_end_candidate=True))
        profiler.record_exclusive(
            "end_window",
            started_at=end_eval_start,
            excluded_stage="feature_match",
            excluded_before_ms=end_eval_match_before,
        )
        if result.decision == KeyframeDecision.ACCEPTED:
            try:
                with profiler.measure("graph_update"):
                    keyframe_id = graph.add_keyframe(
                        frame_id=ekf.frame_id,
                        H_current_to_parent=result.H_current_to_previous,
                        parent_node_id=last_keyframe_node_id,
                    )
                    H_kf_to_global = graph.get_transform(keyframe_id)
                with profiler.measure("l1_inference"):
                    l1_candidates = detector.detect(
                        image=ekf.image,
                        level="L1",
                        frame_id=ekf.frame_id,
                    )
                with profiler.measure("fusion"):
                    fused_candidates = fusion.fuse(l1=l1_candidates, l3=[])
                raw_detections = build_untracked_raw_detections(
                    fused_candidates, keyframe_id, ekf.sharpness_score,
                    ekf.mapping_quality,
                )
                with profiler.measure("projection"):
                    global_detections = [
                        projector.project(
                            detection=rd,
                            H_keyframe_to_global=H_kf_to_global,
                            transform_version=graph.transform_version,
                        )
                        for rd in raw_detections
                    ]
                with profiler.measure("association"):
                    associator.ingest_frame(
                        frame_id=ekf.frame_id,
                        global_detections=global_detections,
                    )
                processed_keyframe_frame_ids.add(ekf.frame_id)
                last_keyframe = ekf
                last_keyframe_frame_id = ekf.frame_id
                last_keyframe_node_id = keyframe_id
            except (ValueError, KeyError) as e:
                print(f"  End KF failed: frame {ekf.frame_id} — {e}")

    # Final review
    with profiler.measure("final_review"):
        associator.final_review()
        associator.map.assign_persistent_ids()

    with profiler.measure("report"):
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
    with profiler.measure("mosaic"):
        generate_global_mosaic(
            video_path=video_path,
            graph=graph,
            objects=display_objects,
            output_dir=output_dir,
        )

    extractor = EvidenceExtractor()
    with profiler.measure("evidence"):
        extractor.extract(video_path, display_objects, output_dir)

    with profiler.measure("session_store"):
        store = SessionStore(output_dir)
        store.create_session(video_path)
        store.save_report(json_report)

    out = Path(output_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    with profiler.measure("report"):
        json_path = out / "reports" / "report.json"
        json_path.write_text(
            json.dumps(json_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        csv_path = out / "reports" / "report.csv"
        csv_path.write_text(csv_report, encoding="utf-8")
    _emit_class_summary_if_enabled(json_report, out, enabled=class_summary)
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
    if performance_enabled:
        performance_snapshot = profiler.save(out, total_frames=fc)
        print("\n" + profiler.format_report(performance_snapshot))

#  "D:\杭州供电段\头戴设备作业工具识别\260814拍摄测试\test.mp4"
def main():
    parser = argparse.ArgumentParser(description="Head Tool Counter - Offline Scan")
    parser.add_argument("--video", default=r"D:\杭州供电段\头戴设备作业工具识别\260814拍摄测试\test.mp4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", default="outputs14")
    parser.add_argument(
        "--performance",
        action="store_true",
        help="输出各处理环节的耗时、调用次数和平均耗时",
    )
    parser.add_argument(
        "--class-summary",
        action="store_true",
        help="输出并保存正式计数对象的类别和数量",
    )
    args = parser.parse_args()
    run_pipeline(
        args.video,
        args.config_dir,
        args.output_dir,
        performance=True if args.performance else None,
        class_summary=args.class_summary,
    )


if __name__ == "__main__":
    main()
