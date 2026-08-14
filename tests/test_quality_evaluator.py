"""QualityEvaluator 单元测试"""
import numpy as np
import cv2

from core.quality_evaluator import InitialKeyframeFallback, QualityEvaluator
from core.types import Frame


def test_sharp_image_passes():
    qe = QualityEvaluator(sharpness_threshold=50.0)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[200:280, 300:340] = 255  # 高对比度边缘
    frame = qe.evaluate(Frame(frame_id=0, timestamp=0.0, image=img))
    assert frame.sharpness_score > 50.0
    assert qe.is_acceptable(frame)


def test_blurry_image_fails():
    qe = QualityEvaluator(sharpness_threshold=50.0)
    img = np.random.randint(100, 110, (480, 640, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (51, 51), 0)
    frame = qe.evaluate(Frame(frame_id=0, timestamp=0.0, image=img))
    assert not qe.is_acceptable(frame)


def test_grayscale_input():
    qe = QualityEvaluator(sharpness_threshold=50.0)
    img = np.zeros((480, 640), dtype=np.uint8)
    img[200:280, 300:340] = 255
    frame = qe.evaluate(Frame(frame_id=0, timestamp=0.0, image=img))
    assert frame.sharpness_score > 50.0


def test_exposure_score_calculated():
    qe = QualityEvaluator()
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    frame = qe.evaluate(Frame(frame_id=0, timestamp=0.0, image=img))
    assert 0.0 <= frame.exposure_score <= 1.0


def test_detection_threshold_is_independent_from_mapping_threshold():
    qe = QualityEvaluator(
        sharpness_threshold=20.0,
        detection_sharpness_threshold=15.0,
    )
    frame = Frame(
        frame_id=1,
        timestamp=0.0,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
    )
    frame.sharpness_score = 17.0

    assert qe.is_acceptable_for_detection(frame)
    assert not qe.is_acceptable(frame)


def test_initial_fallback_waits_then_returns_best_detection_quality_frame():
    fallback = InitialKeyframeFallback(
        min_sharpness=15.0,
        max_interval_frames=30,
    )
    low = Frame(5, 0.0, np.zeros((2, 2, 3), dtype=np.uint8))
    low.sharpness_score = 14.0
    usable = Frame(10, 0.0, np.zeros((2, 2, 3), dtype=np.uint8))
    usable.sharpness_score = 16.0
    best = Frame(20, 0.0, np.zeros((2, 2, 3), dtype=np.uint8))
    best.sharpness_score = 18.0

    fallback.consider(low)
    fallback.consider(usable)
    fallback.consider(best)

    assert fallback.select(29) is None
    assert fallback.select(30) is best


def test_initial_fallback_keeps_waiting_after_interval_without_candidate():
    fallback = InitialKeyframeFallback(
        min_sharpness=15.0,
        max_interval_frames=30,
    )
    low = Frame(30, 0.0, np.zeros((2, 2, 3), dtype=np.uint8))
    low.sharpness_score = 10.0
    later = Frame(45, 0.0, np.zeros((2, 2, 3), dtype=np.uint8))
    later.sharpness_score = 15.0

    fallback.consider(low)
    assert fallback.select(30) is None
    fallback.consider(later)
    assert fallback.select(45) is later


def test_quality_evaluation_uses_configured_half_scale_and_float32_laplacian():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = 255
    evaluator = QualityEvaluator(quality_evaluation_scale=0.5)

    frame = evaluator.evaluate(Frame(1, 0.0, image))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA,
    )
    expected = float(cv2.Laplacian(resized, cv2.CV_32F).var())
    assert frame.sharpness_score == expected


def test_quality_evaluation_scale_must_be_in_valid_range():
    with np.testing.assert_raises(ValueError):
        QualityEvaluator(quality_evaluation_scale=0)
    with np.testing.assert_raises(ValueError):
        QualityEvaluator(quality_evaluation_scale=1.1)
