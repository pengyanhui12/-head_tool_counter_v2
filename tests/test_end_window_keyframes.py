"""尾部关键帧测试 — 正确 parent 链"""
import numpy as np
import pytest

from core.types import Frame
from core.keyframe_selector import KeyframeSelector


class DummyMatcher:
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


def make_frame(frame_id: int, sharpness=80.0):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Frame(frame_id=frame_id, timestamp=frame_id / 30.0, image=img,
                 sharpness_score=sharpness, exposure_score=0.8)


def test_end_keyframes_sorted_ascending():
    """尾部候选按 frame_id 升序排列"""
    selector = KeyframeSelector(
        matcher=DummyMatcher(),
        end_best=2,
    )
    # 先设置 _last_kf_gray 避免 select_end_keyframes 短路
    selector._last_kf_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    frames = [make_frame(100), make_frame(50), make_frame(75)]
    result = selector.select_end_keyframes(frames)
    assert len(result) >= 1
    # 应升序排列
    frame_ids = [f.frame_id for f in result]
    assert frame_ids == sorted(frame_ids)


def test_empty_end_frames():
    """空尾部列表返回空"""
    selector = KeyframeSelector()
    result = selector.select_end_keyframes([])
    assert result == []
