"""证据帧选择应兼顾图像质量和对象全局位置一致性。"""

import numpy as np

from core.evidence_extractor import EvidenceExtractor
from core.types import (
    ConfirmationStatus,
    GlobalDetection,
    GlobalObject,
    VisibilityStatus,
)


def make_detection(
    frame_id: int,
    centroid: tuple[float, float],
    *,
    sharpness: float,
    confidence: float = 1.0,
    mapping_quality: float = 1.0,
    area: float = 900.0,
) -> GlobalDetection:
    """构造用于证据选择测试的全局观测。"""
    x, y = centroid
    return GlobalDetection(
        frame_id=frame_id,
        keyframe_id=frame_id,
        track_id=frame_id,
        projected_corners=np.array([
            [x - 15, y - 15],
            [x + 15, y - 15],
            [x + 15, y + 15],
            [x - 15, y + 15],
        ]),
        projected_center=centroid,
        polygon_centroid=centroid,
        polygon_area=area,
        class_id=0,
        class_name="screwdriver",
        detection_confidence=confidence,
        sharpness=sharpness,
        mapping_quality=mapping_quality,
        edge_quality=1.0,
        size_quality=1.0,
        transform_version=1,
        source="L1",
        bbox_pixels=(0.0, 0.0, 20.0, 20.0),
    )


def make_object(observations: list[GlobalDetection]) -> GlobalObject:
    """构造质心位于目标工具附近的已确认对象。"""
    return GlobalObject(
        provisional_id="P-0019",
        persistent_id="GO-0018",
        class_name="screwdriver",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
        observations=observations,
        centroid_xy=(100.0, 100.0),
    )


def test_select_best_rejects_sharp_but_spatially_inconsistent_observation():
    """高清离群框不能覆盖与对象全局位置一致的代表观测。"""
    representative = make_detection(
        90, (102.0, 99.0), sharpness=180.0
    )
    nearby = make_detection(95, (98.0, 103.0), sharpness=150.0)
    wrong_tool = make_detection(
        98, (175.0, 105.0), sharpness=400.0
    )
    obj = make_object([representative, nearby, wrong_tool])

    selected = EvidenceExtractor.select_best(obj)

    assert selected is representative


def test_select_best_uses_mapping_quality_for_spatially_similar_observations():
    """位置相近时优先选择清晰、置信且映射可靠的观测。"""
    poor_mapping = make_detection(
        10, (100.0, 100.0), sharpness=300.0, mapping_quality=0.2
    )
    reliable = make_detection(
        11, (101.0, 100.0), sharpness=180.0, mapping_quality=0.9
    )
    obj = make_object([poor_mapping, reliable])

    selected = EvidenceExtractor.select_best(obj)

    assert selected is reliable


def test_select_best_falls_back_to_image_quality_without_valid_geometry():
    """缺少有效全局几何时保持原有的图像质量选择行为。"""
    lower = make_detection(1, (float("nan"), 0.0), sharpness=100.0)
    higher = make_detection(2, (float("nan"), 0.0), sharpness=200.0)
    obj = make_object([lower, higher])
    obj.centroid_xy = (float("nan"), float("nan"))

    selected = EvidenceExtractor.select_best(obj)

    assert selected is higher
