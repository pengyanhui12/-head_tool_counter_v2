"""离线扫描 Pipeline——完整流程编排"""
import argparse
import json
import sys
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
from core.quality_evaluator import QualityEvaluator
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
from core.report_generator import ReportGenerator
from core.evidence_extractor import EvidenceExtractor
from core.session_store import SessionStore


def build_raw_detections(
    tracked_detections,
    keyframe_id: int,
    sharpness: float,
    mapping_quality: float,
):
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


def should_run_l2(frame_id: int, interval: int = 3) -> bool:
    return frame_id % interval == 0


def should_trigger_l3(l2_candidates) -> bool:
    return any(
        c.confidence < 0.35 for c in l2_candidates
    )


def select_l3_regions(l2_candidates):
    return [
        tuple(int(v) for v in c.bbox)
        for c in l2_candidates if c.confidence < 0.35
    ]


def run_pipeline(video_path: str, config_dir: str, output_dir: str):
    # Config — resolve relative to project root, not CWD
    _proj_root = Path(__file__).resolve().parent.parent
    cfg_path = (Path(config_dir) if Path(config_dir).is_absolute()
                else _proj_root / config_dir)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config directory not found: {cfg_path}")

    with open(cfg_path / "pipeline.yaml") as f:
        cfg = yaml.safe_load(f)["pipeline"]
    with open(cfg_path / "matcher.yaml") as f:
        mcfg = yaml.safe_load(f)["matcher"]
    with open(cfg_path / "associator.yaml") as f:
        acfg = yaml.safe_load(f)["association"]
    with open(cfg_path / "tracker.yaml") as f:
        tcfg = yaml.safe_load(f)["tracker"]

    l2_interval = cfg.get("l2_interval_frames", 3)
    max_interval = cfg.get("max_keyframe_interval_frames", 30)
    end_window = cfg.get("end_window_frames", 30)

    # Modules
    reader = VideoReader(video_path, max_fps=cfg.get("max_input_fps", 30))
    quality = QualityEvaluator()
    frame_buffer = FrameBuffer(max_size=end_window)
    matcher = FeatureMatcher(
        min_good_matches=mcfg.get("min_good_matches", 20),
        min_inliers=mcfg.get("min_inliers", 30),
        min_inlier_ratio=mcfg.get("min_inlier_ratio", 0.30),
        max_reprojection_error=mcfg.get("max_reprojection_error_px", 3.0),
        min_occupied_quadrants=mcfg.get("min_occupied_quadrants", 3),
        min_inlier_bbox_area_ratio=mcfg.get("min_inlier_bbox_area_ratio", 0.15),
        roi_center_ratio=mcfg.get("roi_center_ratio", 0.70),
    )
    graph = HomographyGraph()
    selector = KeyframeSelector(max_interval=max_interval, end_window_frames=end_window, matcher=matcher)
    detector = Detector(model_path=str(_proj_root / "models" / "best.pt"))
    fusion = DetectionFusion()
    tracker = SimpleDetectionTracker(
        max_missed_detection_frames=tcfg.get("max_missed_detection_frames", 5),
        lost_reactivation_frames=tcfg.get("lost_reactivation_frames", 10),
        min_iou=tcfg.get("min_iou", 0.20),
        iou_weight=tcfg.get("iou_weight", 0.60),
        center_weight=tcfg.get("center_weight", 0.40),
        class_compatibility=acfg.get("class_compatibility", {}),
    )
    projector = GlobalProjector()
    associator = ObjectAssociator(
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
    )
    coverage = CoverageMap(grid_resolution=100)
    status = StatusPanel()

    # State
    last_keyframe: Frame | None = None
    last_keyframe_frame_id: int = -1
    end_window_deque: deque[Frame] = deque(maxlen=end_window)
    fc = 0

    print(f"Processing: {video_path}")
    for frame in reader.read():
        fc += 1
        frame = quality.evaluate(frame)
        if not quality.is_acceptable(frame):
            continue

        frame_buffer.push(frame)
        end_window_deque.append(frame)

        # L2 detection
        l2_candidates = []
        if should_run_l2(fc, l2_interval):
            l2_candidates = detector.detect(
                image=frame.image, level="L2", frame_id=frame.frame_id,
            )

        preview = tracker.preview(l2_candidates)

        trigger_context = KeyframeTriggerContext(
            max_interval_reached=(frame.frame_id - last_keyframe_frame_id >= max_interval),
            l2_new_unmatched_detection=preview.l2_new_unmatched_detection,
            track_quality_drop=preview.track_quality_drop,
            l3_required=should_trigger_l3(l2_candidates),
            l3_regions=select_l3_regions(l2_candidates),
        )

        keyframe_result = selector.evaluate(
            frame=frame,
            previous_keyframe=last_keyframe,
            trigger_context=trigger_context,
        )

        if keyframe_result.decision == KeyframeDecision.ACCEPTED:
            if last_keyframe is None:
                keyframe_id = graph.add_first_keyframe(frame_id=frame.frame_id)
            else:
                keyframe_id = graph.add_keyframe(
                    frame_id=frame.frame_id,
                    H_current_to_previous=keyframe_result.H_current_to_previous,
                )

            H_kf_to_global = graph.get_transform(keyframe_id)

            l1_candidates = detector.detect(
                image=frame.image, level="L1", frame_id=frame.frame_id,
            )

            l3_candidates = []
            if trigger_context.l3_required:
                l3_candidates = detector.detect(
                    image=frame.image, level="L3", frame_id=frame.frame_id,
                    regions=trigger_context.l3_regions,
                )

            fused_candidates = fusion.fuse(l1=l1_candidates, l3=l3_candidates)
            tracked = tracker.update(fused_candidates)

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

            projected_fov = projector.project_frame_corners(
                image_shape=frame.image.shape,
                H_keyframe_to_global=H_kf_to_global,
            )
            coverage.update(frame.frame_id, projected_fov)

            last_keyframe = frame
            last_keyframe_frame_id = frame.frame_id

        elif l2_candidates:
            tracker.update(l2_candidates)

    # Post-video
    print(f"\nProcessed {fc} frames, {graph.num_keyframes} keyframes accepted.")

    # End window: select best 1-2 frames
    end_kfs = selector.select_end_keyframes(list(end_window_deque))
    for ekf in end_kfs:
        result = selector.evaluate(ekf, last_keyframe,
                                   KeyframeTriggerContext(force_end_candidate=True))
        if result.decision == KeyframeDecision.ACCEPTED:
            graph.add_keyframe(ekf.frame_id, result.H_current_to_previous)

    # Final review
    associator.final_review()
    associator.map.assign_persistent_ids()

    gen = ReportGenerator(object_map=associator.map)
    gen.find_evidence_frames(associator.map.get_all())

    # Evidence extraction
    extractor = EvidenceExtractor()
    extractor.extract(video_path, associator.map.get_all(), output_dir)

    # Session save
    store = SessionStore(output_dir)
    store.create_session(video_path)
    store.save_objects(gen.generate_json_report()["objects"])

    out = Path(output_dir)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    json_path = out / "reports" / "report.json"
    json_path.write_text(json.dumps(gen.generate_json_report(), indent=2))
    csv_path = out / "reports" / "report.csv"
    csv_path.write_text(gen.generate_csv_report(associator.map.get_all()))

    total = len(associator.map.get_all())
    confirmed = sum(
        1 for o in associator.map.get_all()
        if o.confirmation_status == ConfirmationStatus.CONFIRMED
    )
    uncertain = sum(
        1 for o in associator.map.get_all()
        if o.confirmation_status == ConfirmationStatus.UNCERTAIN
    )

    print(f"\nDone. {total} objects: {confirmed} CONFIRMED, {uncertain} UNCERTAIN")
    print(f"Reports: {out}/reports/")


def main():
    parser = argparse.ArgumentParser(description="Head Tool Counter - Offline Scan")
    parser.add_argument("--video",default=r"D:\杭州供电段\头戴设备作业工具识别\01公司拍摄数据20260717\测试用\test_first_30s.mp4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_pipeline(args.video, args.config_dir, args.output_dir)


if __name__ == "__main__":
    main()
