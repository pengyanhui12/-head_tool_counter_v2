"""FrameBuffer 单元测试"""
import numpy as np

from core.frame_buffer import FrameBuffer
from core.types import Frame


def _make_frame(frame_id: int) -> Frame:
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Frame(frame_id=frame_id, timestamp=frame_id / 30.0, image=img)


def test_push_and_retrieve():
    buf = FrameBuffer(max_size=30)
    for i in range(10):
        buf.push(_make_frame(i))
    assert buf.size == 10
    frames = buf.get_range(3, 7)
    assert len(frames) == 5
    assert frames[0].frame_id == 3
    assert frames[-1].frame_id == 7


def test_get_by_id():
    buf = FrameBuffer()
    buf.push(_make_frame(42))
    bf = buf.get(42)
    assert bf is not None
    assert bf.frame_id == 42
    assert bf.gray.ndim == 2
    assert buf.get(999) is None


def test_overflow_drops_oldest():
    buf = FrameBuffer(max_size=5)
    for i in range(10):
        buf.push(_make_frame(i))
    assert buf.size == 5
    # 最老的帧 (0-4) 应已被丢弃
    assert buf.get(0) is None
    assert buf.get(5) is not None


def test_bgr_to_gray():
    buf = FrameBuffer()
    buf.push(_make_frame(0))
    bf = buf.get(0)
    assert bf.gray.ndim == 2
    assert bf.gray.shape == (480, 640)
