from pathlib import Path

import cv2
import numpy as np
import pytest

from core.global_mosaic import _sample_nodes, generate_global_mosaic
from core.types import (
    ConfirmationStatus,
    GlobalDetection,
    GlobalObject,
    VisibilityStatus,
)


class _Graph:
    """提供 Mosaic 所需的最小单应图接口。"""

    def __init__(self, nodes):
        self.nodes = nodes
        self.num_keyframes = len(nodes)


class _Capture:
    """返回固定图像并记录资源释放状态。"""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.released = False

    def isOpened(self):
        return True

    def set(self, _property, _value):
        return True

    def read(self):
        return True, self.image.copy()

    def release(self):
        self.released = True


class _CaptureFactory:
    """统计一次 Mosaic 生成过程打开视频的次数。"""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.instances = []

    @property
    def open_count(self):
        return len(self.instances)

    def __call__(self, _video_path):
        capture = _Capture(self.image)
        self.instances.append(capture)
        return capture


def _display_object_with_projected_box(x1, y1, x2, y2):
    """创建会进入可视化输出的确认对象。"""
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float
    )
    detection = GlobalDetection(
        frame_id=0,
        keyframe_id=0,
        track_id=1,
        projected_corners=corners,
        projected_center=((x1 + x2) / 2, (y1 + y2) / 2),
        polygon_centroid=((x1 + x2) / 2, (y1 + y2) / 2),
        polygon_area=abs((x2 - x1) * (y2 - y1)),
        class_id=0,
        class_name="tool",
        detection_confidence=0.9,
        sharpness=100.0,
        mapping_quality=0.9,
        edge_quality=1.0,
        size_quality=1.0,
        transform_version=1,
        source="L1",
    )
    return GlobalObject(
        provisional_id="P-0001",
        persistent_id="GO-0001",
        class_name="tool",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
        observations=[detection],
        centroid_xy=detection.polygon_centroid,
        observation_count=1,
    )


def test_mosaic_opens_video_once_and_warps_each_sampled_node_once(
    tmp_path, monkeypatch
):
    graph = _Graph([(0, 0, np.eye(3)), (1, 10, np.eye(3))])
    obj = _display_object_with_projected_box(20, 20, 120, 80)
    captures = _CaptureFactory(np.full((100, 160, 3), 80, np.uint8))
    warp_calls = []

    monkeypatch.setattr(cv2, "VideoCapture", captures)

    def record_warp(_frame, matrix, size, **kwargs):
        warp_calls.append((matrix.copy(), size, kwargs))
        return np.full((size[1], size[0], 3), 80, np.uint8)

    monkeypatch.setattr(cv2, "warpPerspective", record_warp)
    monkeypatch.setattr(cv2, "imwrite", lambda *_args: True)

    result = generate_global_mosaic("video.mp4", graph, [obj], str(tmp_path))

    assert result == str(tmp_path / "global" / "global_mosaic.jpg")
    assert captures.open_count == 1
    assert len(warp_calls) == 2
    assert all(capture.released for capture in captures.instances)


def test_mosaic_keeps_final_canvas_and_warp_parameters(tmp_path, monkeypatch):
    graph = _Graph([(0, 0, np.eye(3))])
    obj = _display_object_with_projected_box(0, 0, 100, 50)
    captures = _CaptureFactory(np.full((100, 160, 3), 80, np.uint8))
    recorded = {}

    monkeypatch.setattr(cv2, "VideoCapture", captures)

    def record_warp(_frame, _matrix, size, **kwargs):
        # 最终画布参数属于输出视觉契约，优化后不得改变。
        recorded.update(size=size, **kwargs)
        return np.full((size[1], size[0], 3), 80, np.uint8)

    monkeypatch.setattr(cv2, "warpPerspective", record_warp)
    monkeypatch.setattr(cv2, "imwrite", lambda *_args: True)

    generate_global_mosaic("video.mp4", graph, [obj], str(tmp_path))

    assert recorded["size"] == (400, 300)
    assert recorded["flags"] == cv2.INTER_LINEAR
    assert recorded["borderMode"] == cv2.BORDER_TRANSPARENT
    assert recorded["dst"].shape == (300, 400, 3)
    assert np.count_nonzero(recorded["dst"]) == 0


def test_mosaic_matches_legacy_final_pass_for_fixed_input(tmp_path, monkeypatch):
    """固定输入下，新编排应与旧实现真正保存的第二遍画布逐像素一致。"""
    graph = _Graph([(0, 0, np.eye(3))])
    obj = _display_object_with_projected_box(0, 0, 100, 50)
    frame = np.full((100, 160, 3), 80, np.uint8)
    captures = _CaptureFactory(frame)
    written = {}
    monkeypatch.setattr(cv2, "VideoCapture", captures)
    monkeypatch.setattr(
        cv2,
        "imwrite",
        lambda path, image: written.update(path=path, image=image.copy()) or True,
    )

    generate_global_mosaic("video.mp4", graph, [obj], str(tmp_path))

    expected = np.full((300, 400, 3), 40, dtype=np.uint8)
    for x in range(0, 400, 100):
        cv2.line(expected, (x, 0), (x, 300), (60, 60, 60), 1)
    for y in range(0, 300, 100):
        cv2.line(expected, (0, y), (400, y), (60, 60, 60), 1)
    transform = np.array(
        [[1, 0, 50], [0, 1, 50], [0, 0, 1]], dtype=float
    )
    warp_buffer = np.zeros_like(expected)
    warped = cv2.warpPerspective(
        frame,
        transform,
        (400, 300),
        dst=warp_buffer,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    mask = warped.sum(axis=2) > 0
    expected[mask] = cv2.addWeighted(
        expected[mask], 0.75, warped[mask], 0.25, 0
    ).astype(np.uint8)
    pixel_corners = np.array(
        [[50, 50], [150, 50], [150, 100], [50, 100]], dtype=np.int32
    )
    overlay = expected.copy()
    cv2.polylines(overlay, [pixel_corners], True, (0, 255, 0), 1)
    cv2.addWeighted(overlay, 0.3, expected, 0.7, 0, expected)
    cv2.circle(expected, (100, 75), 5, (0, 255, 0), -1)
    cv2.putText(
        expected,
        "GO-0001 tool",
        (106, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1,
    )

    np.testing.assert_array_equal(written["image"], expected)


def test_skip_warp_does_not_open_video_or_call_warp(tmp_path, monkeypatch):
    graph = _Graph([(0, 0, np.eye(3))])
    obj = _display_object_with_projected_box(0, 0, 100, 50)
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *_args: pytest.fail("不应打开视频")
    )
    monkeypatch.setattr(
        cv2,
        "warpPerspective",
        lambda *_args, **_kwargs: pytest.fail("不应执行透视变换"),
    )
    monkeypatch.setattr(cv2, "imwrite", lambda *_args: True)

    result = generate_global_mosaic(
        "video.mp4", graph, [obj], str(tmp_path), skip_warp=True
    )

    assert result == str(Path(tmp_path) / "global" / "global_mosaic.jpg")


def test_mosaic_without_nodes_returns_none(tmp_path):
    assert generate_global_mosaic(
        "video.mp4", _Graph([]), [], str(tmp_path)
    ) is None


def test_mosaic_without_finite_object_points_returns_none(tmp_path):
    graph = _Graph([(0, 0, np.eye(3))])
    obj = _display_object_with_projected_box(np.nan, 0, np.inf, 50)
    obj.centroid_xy = (np.nan, np.inf)

    assert generate_global_mosaic(
        "video.mp4", graph, [obj], str(tmp_path)
    ) is None


def test_mosaic_returns_none_and_releases_when_video_cannot_open(
    tmp_path, monkeypatch
):
    graph = _Graph([(0, 0, np.eye(3))])
    obj = _display_object_with_projected_box(0, 0, 100, 50)

    class ClosedCapture(_Capture):
        def isOpened(self):
            return False

    capture = ClosedCapture(np.zeros((10, 10, 3), dtype=np.uint8))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    result = generate_global_mosaic("video.mp4", graph, [obj], str(tmp_path))

    assert result is None
    assert capture.released


def test_mosaic_skips_unreadable_keyframe_and_writes_remaining_content(
    tmp_path, monkeypatch
):
    graph = _Graph([(0, 0, np.eye(3)), (1, 10, np.eye(3))])
    obj = _display_object_with_projected_box(0, 0, 100, 50)
    frame = np.full((100, 160, 3), 80, dtype=np.uint8)
    warp_calls = []
    written = []

    class SequencedCapture(_Capture):
        def __init__(self):
            super().__init__(frame)
            self.results = [(False, None), (True, frame.copy())]

        def read(self):
            return self.results.pop(0)

    capture = SequencedCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    def record_warp(_frame, _matrix, size, **_kwargs):
        warp_calls.append(size)
        return np.full((size[1], size[0], 3), 80, np.uint8)

    monkeypatch.setattr(cv2, "warpPerspective", record_warp)
    monkeypatch.setattr(
        cv2, "imwrite", lambda path, _image: written.append(path) or True
    )

    result = generate_global_mosaic("video.mp4", graph, [obj], str(tmp_path))

    assert result is not None
    assert len(warp_calls) == 1
    assert written == [result]
    assert capture.released


@pytest.mark.parametrize("invalid_value", [0, -1, True, 1.5])
def test_sample_nodes_rejects_invalid_max_keyframes(invalid_value):
    with pytest.raises(ValueError, match="positive integer"):
        _sample_nodes([(0, 0, np.eye(3))], invalid_value)
