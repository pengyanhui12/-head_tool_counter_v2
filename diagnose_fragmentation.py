"""对象碎片化诊断脚本 — 统计每类工具的正确关联距离分布"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_proj_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_proj_root))

from core.config_loader import ConfigLoader
from core.types import Frame
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
from core.recovery_manager import RecoveryManager, RecoveryState
from core.types import (
    KeyframeDecision, KeyframeTriggerContext, RawDetection,
)


VIDEO_PATH = r"D:\杭州供电段\头戴设备作业工具识别\01公司拍摄数据20260717\测试用\test_cut.mp4"
CONFIG_DIR = _proj_root / "configs"


def main():
    loader = ConfigLoader(CONFIG_DIR)
    cfg = loader.pipeline
    mcfg = loader.matcher
    acfg = loader.associator
    tcfg = loader.tracker

    min_kf_interval = cfg.get("min_keyframe_interval_frames", 5)
    max_interval = cfg.get("max_keyframe_interval_frames", 30)
    end_window = cfg.get("end_window_frames", 30)
    l2_interval = cfg.get("l2_interval_frames", 3)

    reader = VideoReader(VIDEO_PATH, max_fps=cfg.get("max_input_fps", 30))
    quality = QualityEvaluator(
        sharpness_threshold=cfg.get("sharpness_threshold", 20.0),
        dark_pixel_threshold=cfg.get("dark_pixel_threshold", 10),
        bright_pixel_threshold=cfg.get("bright_pixel_threshold", 245),
    )
    matcher = FeatureMatcher(
        min_good_matches=mcfg.get("min_good_matches", 20),
        min_inliers=mcfg.get("min_inliers", 30),
        min_inlier_ratio=mcfg.get("min_inlier_ratio", 0.30),
        max_reprojection_error=mcfg.get("max_reprojection_error_px", 3.0),
        min_occupied_quadrants=mcfg.get("min_occupied_quadrants", 2),
        min_inlier_bbox_area_ratio=mcfg.get("min_inlier_bbox_area_ratio", 0.10),
        roi_center_ratio=mcfg.get("roi_center_ratio", 0.70),
        max_projected_area_ratio=mcfg.get("max_projected_area_ratio", 50.0),
        min_projected_area_ratio=mcfg.get("min_projected_area_ratio", 0.01),
        max_condition_number=mcfg.get("max_condition_number", 500000),
    )
    graph = HomographyGraph()
    selector = KeyframeSelector(
        max_interval=max_interval, end_window_frames=end_window,
        matcher=matcher, min_keyframe_interval_frames=min_kf_interval,
        emergency_keyframe_interval_frames=cfg.get("emergency_keyframe_interval_frames", 2),
    )
    detector = Detector(model_path=str(_proj_root / "models" / "best.pt"))
    fusion = DetectionFusion()
    tracker = SimpleDetectionTracker(
        max_missed_detection_frames=tcfg.get("max_missed_detection_frames", 5),
        lost_reactivation_frames=tcfg.get("lost_reactivation_frames", 10),
        min_iou=tcfg.get("min_iou", 0.20),
        iou_weight=tcfg.get("iou_weight", 0.60),
        center_weight=tcfg.get("center_weight", 0.40),
        class_compatibility=acfg.get("class_compatibility", {}),
        inactive_min_iou=tcfg.get("inactive_min_iou", 0.30),
        inactive_max_center_distance_ratio=tcfg.get("inactive_max_center_distance_ratio", 0.12),
        quality_drop_trigger_ratio=tcfg.get("quality_drop_trigger_ratio", 0.70),
        quality_drop_rearm_ratio=tcfg.get("quality_drop_rearm_ratio", 0.85),
        quality_drop_min_history=tcfg.get("quality_drop_min_history", 5),
    )
    projector = GlobalProjector()
    associator = ObjectAssociator(
        max_position_distance_px=acfg.get("max_position_distance_px", 250.0),
        position_weight=acfg.get("position_weight", 0.55),
        size_weight=acfg.get("size_weight", 0.10),
        class_weight=acfg.get("class_weight", 0.15),
        max_cost=acfg.get("max_cost", 0.75),
        min_observations_confirmed=acfg.get("min_observations_confirmed", 5),
        min_keyframes_confirmed=acfg.get("min_keyframes_confirmed", 3),
        min_top_class_ratio=acfg.get("min_top_class_ratio", 0.60),
        max_votes_per_track=acfg.get("max_votes_per_track", 3),
        class_compatibility=acfg.get("class_compatibility", {}),
        debug_mode=False,
    )
    coverage = CoverageMap(grid_resolution=100)
    recovery_mgr = RecoveryManager(matcher=matcher)

    # State
    last_keyframe = None
    last_keyframe_frame_id = -1
    last_keyframe_node_id = None
    end_window_deque = []
    from collections import deque
    end_window_deque = deque(maxlen=end_window)
    processed_keyframe_frame_ids = set()
    fc = 0

    # ── 距离分布收集 ──
    # per-class: list of (pos_dist, matched_or_not, frame_id, track_id)
    class_distances: dict[str, list[dict]] = defaultdict(list)

    def build_raw_detections(tracked_dets, kf_id, sharpness, mapping_quality):
        raw = []
        for td in tracked_dets:
            c = td.candidate
            x1, y1, x2, y2 = c.bbox
            raw.append(RawDetection(
                frame_id=c.frame_id, keyframe_id=kf_id, track_id=td.track_id,
                bbox=c.bbox, center=((x1+x2)/2, (y1+y2)/2),
                corners=((x1,y1),(x2,y1),(x2,y2),(x1,y2)),
                class_id=c.class_id, class_name=c.class_name,
                confidence=c.confidence, sharpness=sharpness,
                mapping_quality=mapping_quality, source=c.source,
            ))
        return raw

    for frame in reader.read():
        fc += 1
        frame = quality.evaluate(frame)
        if not quality.is_acceptable(frame):
            tracker.advance_frame(frame.frame_id)
            recovery_mgr.cache_frame(frame)
            continue

        end_window_deque.append(frame)
        l2_was_run = (fc % l2_interval == 0)
        l2_candidates = []
        if l2_was_run:
            l2_candidates = detector.detect(image=frame.image, level="L2", frame_id=frame.frame_id)

        preview = tracker.preview(l2_candidates, l2_was_run=l2_was_run)

        trigger_context = KeyframeTriggerContext(
            max_interval_reached=(frame.frame_id - last_keyframe_frame_id >= max_interval),
            l2_new_unmatched_detection=preview.l2_new_unmatched_detection,
            track_quality_drop=preview.track_quality_drop,
            l3_required=(any(c.confidence < 0.35 for c in l2_candidates) if l2_candidates else False),
            l3_regions=[
                tuple(int(v) for v in c.bbox) for c in l2_candidates if c.confidence < 0.35
            ] if l2_candidates else [],
        )

        keyframe_result = selector.evaluate(
            frame=frame, previous_keyframe=last_keyframe, trigger_context=trigger_context,
        )

        if keyframe_result.decision == KeyframeDecision.ACCEPTED:
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

            l1_candidates = detector.detect(image=frame.image, level="L1", frame_id=frame.frame_id)
            l3_candidates = []
            if trigger_context.l3_required:
                l3_candidates = detector.detect(
                    image=frame.image, level="L3", frame_id=frame.frame_id,
                    regions=trigger_context.l3_regions,
                )

            fused_candidates = fusion.fuse(l1=l1_candidates, l3=l3_candidates)
            tracked = tracker.update(fused_candidates, frame_id=frame.frame_id)
            raw_detections = build_raw_detections(tracked, keyframe_id,
                                                   frame.sharpness_score, frame.mapping_quality)
            global_detections = [
                projector.project(detection=rd, H_keyframe_to_global=H_kf_to_global,
                                  transform_version=graph.transform_version)
                for rd in raw_detections
            ]

            # ── 对每个全局检测，计算到所有同类别现有对象的距离 ──
            for gd in global_detections:
                cls = gd.class_name
                existing = [o for o in associator.map.get_all()
                             if o.class_name == cls]
                best_dist = float("inf")
                best_obj_id = None
                for obj in existing:
                    dist = float(np.linalg.norm(
                        np.array(gd.polygon_centroid) - np.array(obj.centroid_xy)
                    ))
                    if dist < best_dist:
                        best_dist = dist
                        best_obj_id = obj.provisional_id
                class_distances[cls].append({
                    "frame_id": frame.frame_id,
                    "track_id": gd.track_id,
                    "best_dist_px": round(best_dist, 1) if best_dist < 1e8 else -1,
                    "has_match": best_dist < associator.max_position_distance * 0.4,
                    "n_existing_same_class": len(existing),
                })

            associator.ingest_frame(frame_id=frame.frame_id,
                                    global_detections=global_detections)
            processed_keyframe_frame_ids.add(frame.frame_id)
            recovery_mgr.reset()
            last_keyframe = frame
            last_keyframe_frame_id = frame.frame_id
            last_keyframe_node_id = keyframe_id
        elif keyframe_result.decision == KeyframeDecision.RECOVERY:
            recovery_result = recovery_mgr.recover(
                current_frame=frame, previous_keyframe=last_keyframe,
                frame_buffer=None, graph=graph, keyframe_images=None,
            )
            if recovery_result.state == RecoveryState.RECOVERED and recovery_result.H_current_to_anchor is not None:
                parent_id = recovery_result.anchor_node_id or last_keyframe_node_id
                try:
                    keyframe_id = graph.add_keyframe(
                        frame_id=frame.frame_id,
                        H_current_to_parent=recovery_result.H_current_to_anchor,
                        parent_node_id=parent_id,
                    )
                    frame.mapping_quality = 0.5
                    H_kf_to_global = graph.get_transform(keyframe_id)
                    l1_candidates = detector.detect(image=frame.image, level="L1", frame_id=frame.frame_id)
                    fused_candidates = fusion.fuse(l1=l1_candidates, l3=[])
                    tracked = tracker.update(fused_candidates, frame_id=frame.frame_id)
                    raw_detections = build_raw_detections(tracked, keyframe_id,
                                                           frame.sharpness_score, frame.mapping_quality)
                    global_detections = [
                        projector.project(detection=rd, H_keyframe_to_global=H_kf_to_global,
                                          transform_version=graph.transform_version)
                        for rd in raw_detections
                    ]
                    associator.ingest_frame(frame_id=frame.frame_id,
                                            global_detections=global_detections)
                    processed_keyframe_frame_ids.add(frame.frame_id)
                    recovery_mgr.reset()
                    last_keyframe = frame
                    last_keyframe_frame_id = frame.frame_id
                    last_keyframe_node_id = keyframe_id
                except (ValueError, KeyError):
                    recovery_mgr.cache_frame(frame)
            else:
                recovery_mgr.cache_frame(frame)
        elif l2_was_run:
            tracker.update(l2_candidates, frame_id=frame.frame_id)

    # ── 打印距离分布 ──
    print("=" * 80)
    print("DISTANCE DISTRIBUTION BY CLASS")
    print("=" * 80)
    import statistics

    for cls_name in sorted(class_distances.keys()):
        dists = class_distances[cls_name]
        matched = [d for d in dists if d["has_match"]]
        unmatched = [d for d in dists if not d["has_match"] and d["best_dist_px"] >= 0]

        all_dists = [d["best_dist_px"] for d in dists if d["best_dist_px"] >= 0]
        matched_dists = [d["best_dist_px"] for d in matched]
        unmatched_dists = [d["best_dist_px"] for d in unmatched]

        print(f"\n--- {cls_name} ---")
        print(f"  Total observations: {len(dists)}")
        print(f"  Matched: {len(matched)}, Unmatched: {len(unmatched)}")
        if all_dists:
            print(f"  All dists: min={min(all_dists):.0f} max={max(all_dists):.0f} "
                  f"mean={statistics.mean(all_dists):.0f} median={statistics.median(all_dists):.0f} "
                  f"p50={np.percentile(all_dists, 50):.0f} p75={np.percentile(all_dists, 75):.0f} "
                  f"p90={np.percentile(all_dists, 90):.0f} p95={np.percentile(all_dists, 95):.0f}")
        if matched_dists:
            print(f"  Matched:    min={min(matched_dists):.0f} max={max(matched_dists):.0f} "
                  f"mean={statistics.mean(matched_dists):.0f} median={statistics.median(matched_dists):.0f}")
        if unmatched_dists:
            print(f"  Unmatched:  min={min(unmatched_dists):.0f} max={max(unmatched_dists):.0f} "
                  f"mean={statistics.mean(unmatched_dists):.0f} median={statistics.median(unmatched_dists):.0f}")

    # ── 打印最终对象碎片化指标 ──
    associator.final_review()
    associator.map.assign_persistent_ids()

    print("\n" + "=" * 80)
    print("FRAGMENTATION SUMMARY")
    print("=" * 80)
    objects = associator.map.get_all()
    reportable = associator.get_reportable_objects()
    by_class = defaultdict(list)
    for o in reportable:
        by_class[o.class_name].append(o)

    print(f"\nTotal reportable objects: {len(reportable)}")
    for cls_name, objs in sorted(by_class.items()):
        n_tracks = [len(o.track_ids) for o in objs]
        dists_between = []
        for i in range(len(objs)):
            for j in range(i+1, len(objs)):
                d = float(np.linalg.norm(
                    np.array(objs[i].centroid_xy) - np.array(objs[j].centroid_xy)
                ))
                dists_between.append(d)
        shared_track_pairs = sum(
            1 for i in range(len(objs)) for j in range(i+1, len(objs))
            if objs[i].track_ids & objs[j].track_ids
        )
        print(f"  {cls_name}: {len(objs)} objects, "
              f"tracks/obj avg={sum(n_tracks)/len(n_tracks):.1f} max={max(n_tracks)}, "
              f"inter-object dist min={min(dists_between):.0f}px" if dists_between else f"  {cls_name}: {len(objs)} objects (1 only)",
              f"shared_track_pairs={shared_track_pairs}")


if __name__ == "__main__":
    main()
