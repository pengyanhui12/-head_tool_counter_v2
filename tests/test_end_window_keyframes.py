"""尾部关键帧测试 — 正确 parent 链"""
import numpy as np
import pytest

from core.types import Frame
from core.keyframe_selector import KeyframeSelector


class DummyMatcher:
    def __init__(self):
        self.calls = 0
        self.seen_images = []

    def match(self, src, dst):
        self.calls += 1
        self.seen_images.append(src)
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


def test_end_window_prefilter_matches_six_temporally_distributed_frames():
    matcher = DummyMatcher()
    selector = KeyframeSelector(
        matcher=matcher,
        end_best=2,
        end_window_match_candidates=6,
    )
    selector._last_kf_gray = np.zeros((8, 8), dtype=np.uint8)
    frames = [make_frame(frame_id, sharpness=float(frame_id))
              for frame_id in range(30)]
    image_to_frame = {id(frame.image): frame.frame_id for frame in frames}

    selector.select_end_keyframes(frames)

    selected_for_matching = [
        image_to_frame[id(image)] for image in matcher.seen_images
    ]
    assert matcher.calls == 6
    assert selected_for_matching == [4, 9, 14, 19, 24, 29]


def test_end_window_prefilter_prefers_later_frame_on_quality_tie():
    matcher = DummyMatcher()
    selector = KeyframeSelector(
        matcher=matcher,
        end_window_match_candidates=2,
    )
    selector._last_kf_gray = np.zeros((8, 8), dtype=np.uint8)
    frames = [make_frame(frame_id, sharpness=80.0) for frame_id in range(4)]
    image_to_frame = {id(frame.image): frame.frame_id for frame in frames}

    selector.select_end_keyframes(frames)

    selected_for_matching = [
        image_to_frame[id(image)] for image in matcher.seen_images
    ]
    assert selected_for_matching == [1, 3]


@pytest.mark.parametrize("candidate_count", [0, 30, 40])
def test_end_window_prefilter_can_fall_back_to_full_matching(candidate_count):
    matcher = DummyMatcher()
    selector = KeyframeSelector(
        matcher=matcher,
        end_window_match_candidates=candidate_count,
    )
    selector._last_kf_gray = np.zeros((8, 8), dtype=np.uint8)
    frames = [make_frame(frame_id) for frame_id in range(30)]

    selector.select_end_keyframes(frames)

    assert matcher.calls == 30
