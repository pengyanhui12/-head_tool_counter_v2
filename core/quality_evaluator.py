"""图像质量评估器——清晰度 + 曝光（欠曝/过曝分离）"""
import cv2
import numpy as np

from core.types import Frame


class QualityEvaluator:
    def __init__(
        self,
        sharpness_threshold: float = 50.0,
        dark_pixel_threshold: int = 10,
        bright_pixel_threshold: int = 245,
        underexposed_ratio: float = 0.5,
        overexposed_ratio: float = 0.3,
    ):
        self.sharpness_threshold = sharpness_threshold
        self.dark_pixel_threshold = dark_pixel_threshold
        self.bright_pixel_threshold = bright_pixel_threshold
        self.underexposed_ratio = underexposed_ratio
        self.overexposed_ratio = overexposed_ratio

    def evaluate(self, frame: Frame) -> Frame:
        image = frame.image
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        frame.sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 曝光：分离欠曝和过曝
        total_pixels = gray.size
        underexposed_pct = float(np.sum(gray < self.dark_pixel_threshold) / total_pixels)
        overexposed_pct = float(np.sum(gray > self.bright_pixel_threshold) / total_pixels)
        frame.exposure_score = float(np.clip(1.0 - underexposed_pct - overexposed_pct, 0.0, 1.0))

        return frame

    def is_acceptable(self, frame: Frame) -> bool:
        """质量是否可接受（用于建图和检测）。"""
        return frame.sharpness_score >= self.sharpness_threshold

    def is_acceptable_for_mapping(self, frame: Frame) -> bool:
        """质量是否适合建图（更严格）。"""
        if not self.is_acceptable(frame):
            return False
        # 额外要求曝光不过差
        if frame.exposure_score < 0.3:
            return False
        return True

    def is_acceptable_for_detection(self, frame: Frame) -> bool:
        """质量是否适合运行检测（较宽松）。"""
        return frame.sharpness_score >= self.sharpness_threshold * 0.5
