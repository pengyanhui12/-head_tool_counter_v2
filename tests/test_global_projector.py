"""GlobalProjector 单元测试"""
import numpy as np

from core.global_projector import GlobalProjector
from core.types import RawDetection


def _raw_det(bbox, class_id=0, class_name="wrench", keyframe_id=0, track_id=0):
    x1, y1, x2, y2 = bbox
    return RawDetection(
        frame_id=0, keyframe_id=keyframe_id, track_id=track_id,
        bbox=bbox,
        center=((x1+x2)/2, (y1+y2)/2),
        corners=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        class_id=class_id, class_name=class_name, confidence=0.9,
        sharpness=100.0, mapping_quality=0.8, source="L1",
    )


def test_project_identity():
    H = np.eye(3)
    det = _raw_det((100, 100, 200, 200))
    gd = GlobalProjector.project(det, H, 0)
    assert abs(gd.polygon_centroid[0] - 150.0) < 0.1
    assert abs(gd.polygon_centroid[1] - 150.0) < 0.1
    assert gd.polygon_area > 0


def test_project_translation():
    H = np.array([[1, 0, 50], [0, 1, 30], [0, 0, 1]], dtype=float)
    det = _raw_det((100, 100, 200, 200))
    gd = GlobalProjector.project(det, H, 0)
    assert abs(gd.polygon_centroid[0] - 200.0) < 0.1
    assert abs(gd.polygon_centroid[1] - 180.0) < 0.1


def test_project_frame_corners():
    H = np.eye(3)
    poly = GlobalProjector.project_frame_corners((480, 640), H)
    assert poly.area > 0
    # should be approx 640*480 = 307200
    assert abs(poly.area - 307200) < 1000
