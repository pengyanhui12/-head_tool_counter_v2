"""帧缓冲区——灰度图缓存，内存受控"""
from collections import deque
from collections.abc import Iterator

import cv2

from core.types import Frame, BufferedFrame


class FrameBuffer:
    def __init__(self, max_size: int = 60):
        self._buf: deque[BufferedFrame] = deque(maxlen=max_size)

    def push(self, frame: Frame) -> None:
        image = frame.image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        bf = BufferedFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            gray=gray,
            sharpness_score=frame.sharpness_score,
            source_frame_id=frame.frame_id,
        )
        self._buf.append(bf)

    def get_range(self, start_id: int, end_id: int) -> list[BufferedFrame]:
        return [f for f in self._buf if start_id <= f.frame_id <= end_id]

    def get(self, frame_id: int) -> BufferedFrame | None:
        for f in self._buf:
            if f.frame_id == frame_id:
                return f
        return None

    def __iter__(self) -> Iterator[BufferedFrame]:
        return iter(self._buf)

    @property
    def size(self) -> int:
        return len(self._buf)
