"""图像质量评估器——清晰度 + 曝光（欠曝/过曝分离）"""
import cv2
import numpy as np

from core.types import Frame


class InitialKeyframeFallback:
    """在首个正常关键帧出现前保留最佳检测级帧。"""

    def __init__(self, min_sharpness: float, max_interval_frames: int):
        self.min_sharpness = min_sharpness
        self.max_interval_frames = max_interval_frames
        self._best: Frame | None = None

    def consider(self, frame: Frame) -> None:
        if frame.sharpness_score < self.min_sharpness:
            return
        if self._best is None or frame.sharpness_score > self._best.sharpness_score:
            self._best = frame

    def select(self, current_frame_id: int) -> Frame | None:
        if current_frame_id < self.max_interval_frames:
            return None
        return self._best

    def clear(self) -> None:
        self._best = None


class QualityEvaluator:
    def __init__(
        self,
        sharpness_threshold: float = 20.0,
        dark_pixel_threshold: int = 10,
        bright_pixel_threshold: int = 245,
        underexposed_ratio: float = 0.5,
        overexposed_ratio: float = 0.3,
        detection_sharpness_threshold: float | None = None,
        quality_evaluation_scale: float = 1.0,
    ):
        if not 0.0 < quality_evaluation_scale <= 1.0:
            raise ValueError("quality_evaluation_scale must be in (0, 1]")
        self.sharpness_threshold = sharpness_threshold
        self.dark_pixel_threshold = dark_pixel_threshold
        self.bright_pixel_threshold = bright_pixel_threshold
        self.underexposed_ratio = underexposed_ratio
        self.overexposed_ratio = overexposed_ratio
        self.detection_sharpness_threshold = (
            sharpness_threshold * 0.5
            if detection_sharpness_threshold is None
            else detection_sharpness_threshold
        )
        self.quality_evaluation_scale = quality_evaluation_scale

    def evaluate(self, frame: Frame) -> Frame:
        image = frame.image
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        quality_gray = gray
        if self.quality_evaluation_scale < 1.0:
            quality_gray = cv2.resize(
                gray,
                None,
                fx=self.quality_evaluation_scale,
                fy=self.quality_evaluation_scale,
                interpolation=cv2.INTER_AREA,
            )

        frame.sharpness_score = float(
            cv2.Laplacian(quality_gray, cv2.CV_32F).var()
        )

        # 曝光：分离欠曝和过曝
        total_pixels = quality_gray.size
        underexposed_pct = float(
            np.sum(quality_gray < self.dark_pixel_threshold) / total_pixels
        )
        overexposed_pct = float(
            np.sum(quality_gray > self.bright_pixel_threshold) / total_pixels
        )
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
        return frame.sharpness_score >= self.detection_sharpness_threshold
