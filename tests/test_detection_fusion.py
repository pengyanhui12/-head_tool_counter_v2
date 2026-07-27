"""DetectionFusion 单元测试"""
from core.detection_fusion import DetectionFusion
from core.types import DetectionCandidate


def _cand(frame_id: int, class_id: int, class_name: str,
           bbox: tuple, confidence: float = 0.9) -> DetectionCandidate:
    return DetectionCandidate(
        frame_id=frame_id, bbox=bbox, class_id=class_id,
        class_name=class_name, confidence=confidence,
        source="L1", image_width=640, image_height=480,
    )


def test_no_l3_returns_l1():
    l1 = [_cand(0, 0, "wrench", (100, 100, 200, 200))]
    result = DetectionFusion().fuse(l1, [])
    assert len(result) == 1
    assert result[0].class_name == "wrench"


def test_fuse_removes_duplicate():
    l1 = [_cand(0, 0, "wrench", (100, 100, 200, 200), 0.9)]
    l3 = [_cand(0, 0, "wrench", (105, 105, 195, 195), 0.8)]
    result = DetectionFusion(iou_threshold=0.5).fuse(l1, l3)
    assert len(result) == 1


def test_different_classes_kept():
    l1 = [_cand(0, 0, "wrench", (100, 100, 200, 200))]
    l3 = [_cand(0, 1, "plier",  (300, 300, 400, 400))]
    result = DetectionFusion().fuse(l1, l3)
    assert len(result) == 2
