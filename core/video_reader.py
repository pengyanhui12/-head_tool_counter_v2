"""视频读取器——时间累积采样，避免整数截断"""
from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from core.types import Frame


class VideoReader:
    def __init__(self, path: str, max_fps: float = 30.0):
        if max_fps <= 0:
            raise ValueError("max_fps must be positive")
        self.path = path
        self.max_fps = float(max_fps)

    def read(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self.path}")

        try:
            src_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(src_fps) or src_fps <= 0:
                src_fps = 30.0

            output_interval = 1.0 / self.max_fps
            next_output_time = 0.0
            source_frame_id = 0

            while True:
                ok, image = cap.read()
                if not ok:
                    break

                timestamp = source_frame_id / src_fps
                if timestamp + 1e-9 >= next_output_time:
                    yield Frame(
                        frame_id=source_frame_id,
                        timestamp=timestamp,
                        image=image,
                    )
                    next_output_time += output_interval

                source_frame_id += 1
        finally:
            cap.release()
