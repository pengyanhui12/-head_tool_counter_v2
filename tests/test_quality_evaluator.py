"""QualityEvaluator 单元测试"""
import numpy as np
import cv2

from core.quality_evaluator import QualityEvaluator
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
