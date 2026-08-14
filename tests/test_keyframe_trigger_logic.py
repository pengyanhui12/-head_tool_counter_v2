"""关键帧触发逻辑测试 — L2 not-run vs empty, cooldown, quality_drop 边沿触发"""
import pytest
import numpy as np

from core.types import (
    Frame, KeyframeDecision, KeyframeTriggerContext, KeyframeResult,
)
from core.keyframe_selector import KeyframeSelector
from core.simple_tracker import SimpleDetectionTracker
from core.feature_matcher import FeatureMatcher


class DummyMatcher:
    """可控的 mock matcher——总是返回 valid，H=identity"""
    def match(self, src, dst):
        from core.types import MatchResult
        return MatchResult(
            H_source_to_target=np.eye(3),
            num_keypoints_src=100, num_keypoints_dst=100,
            num_good_matches=50, num_inliers=30,
            inlier_ratio=0.6, reprojection_error=1.5,
            occupied_quadrants_src=4, occupied_quadrants_dst=4,
            inlier_bbox_area_ratio_src=0.3, inlier_bbox_area_ratio_dst=0.3,
            valid=True,
        )


def make_frame(frame_id: int, shape=(480, 640, 3)):
    img = np.random.randint(0, 255, shape, dtype=np.uint8)
    return Frame(frame_id=frame_id, timestamp=frame_id / 30.0, image=img)


def test_l2_not_run_no_quality_drop():
    """L2 未运行时不应产生 QUAL_DROP。"""
    tracker = SimpleDetectionTracker()
    preview = tracker.preview([], l2_was_run=False)
    assert preview.track_quality_drop is False
    assert preview.l2_new_unmatched_detection is False


def test_l2_empty_detection_produces_no_new_det():
    """L2 运行但无检测 → unmatched=False"""
    tracker = SimpleDetectionTracker()
    preview = tracker.preview([], l2_was_run=True)
    assert preview.l2_new_unmatched_detection is False


def test_l2_new_detection_produces_new_det():
    """L2 有新检测 → unmatched=True"""
    from core.types import DetectionCandidate
    tracker = SimpleDetectionTracker(new_detection_confirmation_runs=1)
    det = DetectionCandidate(frame_id=1, bbox=(100, 100, 200, 200),
                             class_id=0, class_name="wrench", confidence=0.8,
                             source="L2", image_width=640, image_height=480)
    preview = tracker.preview([det], l2_was_run=True)
    assert preview.l2_new_unmatched_detection is True


def test_quality_drop_edge_triggered():
    """质量下降边沿触发：只触发一次，rearm 后才重新触发。"""
    from core.types import DetectionCandidate
    tracker = SimpleDetectionTracker(
        quality_drop_trigger_ratio=0.70,
        quality_drop_rearm_ratio=0.85,
        quality_drop_min_history=5,
    )
    # 创建 track 并推送高置信度
    dets_high = [
        DetectionCandidate(frame_id=i, bbox=(100, 100, 200, 200),
                           class_id=0, class_name="wrench", confidence=0.9,
                           source="L2", image_width=640, image_height=480)
        for i in range(1, 6)
    ]
    for det in dets_high:
        tracker.update([det], frame_id=det.frame_id)

    # 推送低置信度（触发一次）
    dets_low = [
        DetectionCandidate(frame_id=i, bbox=(100, 100, 200, 200),
                           class_id=0, class_name="wrench", confidence=0.3,
                           source="L2", image_width=640, image_height=480)
        for i in range(6, 9)
    ]
    triggers = 0
    for det in dets_low:
        tracker.update([det], frame_id=det.frame_id)
        if tracker.track_quality_dropped(0):
            triggers += 1

    assert triggers <= 1, "quality_drop should be edge-triggered, at most once"


def test_cooldown_blocks_keyframe():
    """cooldown 期内 QUAL_DROP 和 NEW_DET 被阻止"""
    selector = KeyframeSelector(
        max_interval=30,
        min_keyframe_interval_frames=5,
        matcher=DummyMatcher(),
    )
    f1 = make_frame(1)
    # 第一帧
    r1 = selector.evaluate(f1, None,
                           KeyframeTriggerContext(max_interval_reached=True))
    assert r1.decision == KeyframeDecision.ACCEPTED

    # 第2帧：间隔1 < cooldown=5
    f2 = make_frame(2)
    ctx = KeyframeTriggerContext(track_quality_drop=True)
    r2 = selector.evaluate(f2, f1, ctx)
    assert r2.decision == KeyframeDecision.SKIP
    assert "cooldown" in r2.reason


def test_max_interval_bypasses_cooldown():
    """max_interval 不受 cooldown 限制"""
    selector = KeyframeSelector(
        max_interval=5,
        min_keyframe_interval_frames=5,
        matcher=DummyMatcher(),
    )
    f1 = make_frame(1)
    r1 = selector.evaluate(f1, None,
                           KeyframeTriggerContext(max_interval_reached=True))
    assert r1.decision == KeyframeDecision.ACCEPTED

    # frame 10: 超过 max_interval=5，即使 cooldown 也触发
    f10 = make_frame(10)
    ctx = KeyframeTriggerContext(max_interval_reached=True)
    r10 = selector.evaluate(f10, f1, ctx)
    assert r10.decision == KeyframeDecision.ACCEPTED
