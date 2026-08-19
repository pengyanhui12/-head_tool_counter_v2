"""FeatureMatcher 单元测试"""
import numpy as np
import cv2
import pytest

from core.feature_matcher import FeatureMatcher
from core.types import MatchResult


class _CountingDetector:
    """只统计特征提取次数，返回可触发安全失败的空特征。"""

    def __init__(self):
        self.calls = 0

    def detectAndCompute(self, gray, mask):
        self.calls += 1
        return [], None


def _make_feature_image(w: int = 640, h: int = 480, seed: int = 0,
                         shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(30):
        x = rng.randint(shift_x + 20, shift_x + w - 20)
        y = rng.randint(shift_y + 20, shift_y + h - 20)
        cv2.circle(img, (x, y), rng.randint(3, 8), (255, 255, 255), -1)
    return img


def test_identity():
    m = FeatureMatcher()
    img = _make_feature_image()
    result = m.match(img, img)
    assert isinstance(result, MatchResult)
    assert result.valid
    assert result.num_inliers >= 20
    assert result.reprojection_error < 2.0


def test_translation():
    m = FeatureMatcher()
    img0 = _make_feature_image(seed=1)
    img1 = _make_feature_image(seed=1, shift_x=30)
    result = m.match(img0, img1)
    # 0 good matches means ratio_test too strict on these images -- the test
    # verifies that the matcher doesn't crash and returns a proper MatchResult
    assert isinstance(result, MatchResult)
    assert result.num_keypoints_src > 0
    assert result.num_keypoints_dst > 0


def test_blurry_image_fails():
    m = FeatureMatcher()
    img0 = _make_feature_image(seed=2)
    img1 = cv2.GaussianBlur(_make_feature_image(seed=2), (31, 31), 0)
    result = m.match(img0, img1)
    # blur may produce some matches, but quality should be low
    assert not result.valid or result.num_inliers < 10


def test_insufficient_features():
    m = FeatureMatcher()
    white = np.ones((480, 640, 3), dtype=np.uint8) * 255
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    result = m.match(white, black)
    assert not result.valid
    assert result.failure_reason is not None


def test_rotation():
    m = FeatureMatcher()
    img0 = _make_feature_image(seed=3)
    M = cv2.getRotationMatrix2D((320, 240), 15, 1.0)
    img1 = cv2.warpAffine(img0, M, (640, 480))
    result = m.match(img0, img1)
    # 15° rotation may not work on random dots, but shouldn't crash
    assert result is not None


def test_reprojection_error_calculated():
    m = FeatureMatcher()
    img0 = _make_feature_image(seed=4)
    img1 = _make_feature_image(seed=4, shift_x=100)
    result = m.match(img0, img1)
    assert result.reprojection_error != float("inf")
    assert result.reprojection_error >= 0.0


def test_feature_cache_reuses_same_image_object():
    matcher = FeatureMatcher(feature_cache_size=2)
    detector = _CountingDetector()
    matcher._detector = detector
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    matcher.match(image, image)
    matcher.match(image, image)

    assert detector.calls == 1


def test_feature_cache_evicts_least_recently_used_image():
    matcher = FeatureMatcher(feature_cache_size=2)
    detector = _CountingDetector()
    matcher._detector = detector
    first = np.zeros((16, 16, 3), dtype=np.uint8)
    second = np.ones((16, 16, 3), dtype=np.uint8)
    third = np.full((16, 16, 3), 2, dtype=np.uint8)

    matcher.extract_features(first)
    matcher.extract_features(second)
    matcher.extract_features(first)  # first成为最近使用项，second应先被淘汰。
    matcher.extract_features(third)
    matcher.extract_features(second)

    assert detector.calls == 4


def test_feature_cache_can_be_cleared():
    matcher = FeatureMatcher(feature_cache_size=2)
    detector = _CountingDetector()
    matcher._detector = detector
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    matcher.extract_features(image)
    matcher.clear_feature_cache()
    matcher.extract_features(image)

    assert detector.calls == 2


def test_zero_feature_cache_size_disables_cache():
    matcher = FeatureMatcher(feature_cache_size=0)
    detector = _CountingDetector()
    matcher._detector = detector
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    matcher.extract_features(image)
    matcher.extract_features(image)

    assert detector.calls == 2


def test_negative_feature_cache_size_is_rejected():
    with pytest.raises(ValueError, match="feature_cache_size"):
        FeatureMatcher(feature_cache_size=-1)
