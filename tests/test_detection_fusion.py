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


def test_nms_applies_when_only_l1_is_present():
    l1 = [
        _cand(0, 0, "wrench", (100, 100, 200, 200), 0.9),
        _cand(0, 0, "wrench", (105, 105, 195, 195), 0.8),
    ]

    result = DetectionFusion(iou_threshold=0.5).fuse(l1, [])

    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_center_dedup_removes_overlapping_low_iou_duplicate():
    l1 = [
        _cand(0, 0, "wrench", (100, 100, 180, 180), 0.9),
        _cand(0, 0, "wrench", (125, 100, 205, 180), 0.8),
    ]

    result = DetectionFusion(
        iou_threshold=0.65,
        center_merge_distance_px=40,
        center_merge_min_ios=0.5,
    ).fuse(l1, [])

    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_center_dedup_keeps_nearby_non_overlapping_instances():
    l1 = [
        _cand(0, 0, "screwdriver", (100, 100, 110, 130), 0.9),
        _cand(0, 0, "screwdriver", (125, 100, 135, 130), 0.8),
    ]

    result = DetectionFusion(
        center_merge_distance_px=40,
        center_merge_min_ios=0.3,
    ).fuse(l1, [])

    assert len(result) == 2


def test_center_distance_supports_per_class_override():
    l1 = [
        _cand(0, 0, "pliers", (100, 100, 160, 160), 0.9),
        _cand(0, 0, "pliers", (120, 100, 180, 160), 0.8),
    ]

    result = DetectionFusion(
        center_merge_distance_px=40,
        per_class_center_merge_distances={"pliers": 10},
    ).fuse(l1, [])

    assert len(result) == 2
