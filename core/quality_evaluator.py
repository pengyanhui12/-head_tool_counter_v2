"""图像质量评估器——清晰度 + 曝光"""
import cv2
import numpy as np

from core.types import Frame


class QualityEvaluator:
    def __init__(self, sharpness_threshold: float = 50.0):
        self.sharpness_threshold = sharpness_threshold

    def evaluate(self, frame: Frame) -> Frame:
        image = frame.image
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        frame.sharpness_score = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )
        frame.exposure_score = float(
            np.clip(gray.mean() / 127.5, 0.0, 1.0)
        )
        return frame

    def is_acceptable(self, frame: Frame) -> bool:
        return frame.sharpness_score >= self.sharpness_threshold
