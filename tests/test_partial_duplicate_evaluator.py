import pickle

import numpy as np
import pytest

from core.partial_duplicate_evaluator import (
    PartialDuplicateEvaluator,
)
from core.types import (
    ConfirmationStatus,
    GlobalDetection,
    GlobalObject,
    VisibilityStatus,
)


def make_detection(
    *,
    frame_id: int,
    keyframe_id: int,
    bbox_pixels=(0.0, 0.0, 100.0, 100.0),
    projected_corners=None,
    mapping_quality: float = 0.9,
    area: float = 10_000.0,
) -> GlobalDetection:
    if projected_corners is None:
        projected_corners = np.array(
            [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
        )
    return GlobalDetection(
        frame_id=frame_id,
        keyframe_id=keyframe_id,
        track_id=keyframe_id,
        projected_corners=np.asarray(projected_corners, dtype=float),
        projected_center=(50.0, 50.0),
        polygon_centroid=(50.0, 50.0),
        polygon_area=area,
        class_id=0,
        class_name="wrench",
        detection_confidence=0.8,
        sharpness=80.0,
        mapping_quality=mapping_quality,
        edge_quality=1.0,
        size_quality=1.0,
        transform_version=1,
        source="L1",
        bbox_pixels=bbox_pixels,
    )


def make_object(
    object_id: str,
    *,
    status: ConfirmationStatus,
    frame_id: int,
    bbox_pixels=(0.0, 0.0, 100.0, 100.0),
    projected_corners=None,
    mapping_quality: float = 0.9,
    centroid=(50.0, 50.0),
    area: float = 10_000.0,
    area_range=None,
    observation_count: int = 1,
    keyframe_ids=None,
    class_name: str = "wrench",
) -> GlobalObject:
    detection = make_detection(
        frame_id=frame_id,
        keyframe_id=frame_id,
        bbox_pixels=bbox_pixels,
        projected_corners=projected_corners,
        mapping_quality=mapping_quality,
        area=area,
    )
    detection.class_name = class_name
    return GlobalObject(
        provisional_id=object_id,
        persistent_id=None,
        class_name=class_name,
        confirmation_status=status,
        visibility_status=VisibilityStatus.ACTIVE,
        observations=[detection],
        centroid_xy=centroid,
        area_range=area_range or (area, area),
        keyframe_ids=set(keyframe_ids or {frame_id}),
        observation_count=observation_count,
    )


def test_unique_contained_fragment_is_attributed():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "attributed"
    assert decision.candidate_id == "C-1"
    assert decision.candidate_ids == ("C-1",)
    assert decision.containment_score == 1.0


def test_same_frame_independent_same_class_objects_are_blocked():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(60.0, 0.0, 100.0, 40.0),
        centroid=(80.0, 20.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=10,
        bbox_pixels=(0.0, 0.0, 40.0, 40.0),
    )

    decision = PartialDuplicateEvaluator().evaluate(
        tentative, [confirmed], {frozenset(("T-1", "C-1"))}
    )

    assert decision.decision == "no_match"
    assert decision.reason == "independent_co_occurrence"


def test_distance_without_containment_is_rejected():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(110.0, 0.0, 150.0, 40.0),
        centroid=(55.0, 50.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "insufficient_containment"


def test_low_mapping_quality_blocks_cross_frame_attribution():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        projected_corners=[[20, 20], [60, 20], [60, 60], [20, 60]],
        mapping_quality=0.4,
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=11
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "low_mapping_quality"


def test_non_finite_mapping_quality_does_not_meet_cross_frame_threshold():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        projected_corners=[[20, 20], [60, 20], [60, 60], [20, 60]],
        mapping_quality=float("nan"),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=11
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "low_mapping_quality"


def test_reliable_cross_frame_global_polygons_can_attribute():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(500.0, 500.0, 540.0, 540.0),
        projected_corners=[[20, 20], [60, 20], [60, 60], [20, 60]],
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=11,
        bbox_pixels=(0.0, 0.0, 100.0, 100.0),
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "attributed"
    assert decision.candidate_id == "C-1"
    assert decision.containment_score == 1.0


def test_adjacent_frame_raw_boxes_alone_are_not_compared():
    empty_polygon = np.empty((0, 2), dtype=float)
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        projected_corners=empty_polygon,
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=11,
        projected_corners=empty_polygon,
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "no_comparable_geometry"


def test_similarly_scored_candidates_are_ambiguous():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed_a = make_object(
        "C-1",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=10,
        centroid=(40.0, 40.0),
    )
    confirmed_b = make_object(
        "C-2",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=10,
        centroid=(45.0, 40.0),
    )

    decision = PartialDuplicateEvaluator().evaluate(
        tentative, [confirmed_b, confirmed_a], set()
    )

    assert decision.decision == "ambiguous"
    assert decision.candidate_id is None
    assert decision.candidate_ids == ("C-1", "C-2")
    assert decision.reason == "candidate_margin_below_threshold"


def test_non_tentative_input_is_rejected():
    not_tentative = make_object(
        "C-0", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )

    decision = PartialDuplicateEvaluator().evaluate(not_tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "not_tentative"


def test_full_confirmation_strength_tentative_is_rejected():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
        observation_count=3,
        keyframe_ids={10, 11},
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "sufficient_confirmation_evidence"


@pytest.mark.parametrize(
    ("observation_count", "keyframe_ids"),
    [
        (3, {10}),
        (1, {10, 11}),
    ],
)
def test_tentative_remains_eligible_when_either_evidence_threshold_is_unmet(
    observation_count, keyframe_ids
):
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
        observation_count=observation_count,
        keyframe_ids=keyframe_ids,
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "attributed"


def test_non_finite_same_frame_box_is_not_comparable_geometry():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, float("inf"), 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "no_comparable_geometry"


def test_non_finite_centroid_cannot_bypass_distance_guard():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(float("nan"), 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "distance_exceeded"


def test_non_finite_area_cannot_bypass_scale_or_distance_guards():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=10,
        area_range=(10_000.0, float("inf")),
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "no_match"
    assert decision.reason == "distance_exceeded"


def test_only_same_class_confirmed_candidates_are_considered():
    tentative = make_object(
        "T-1", status=ConfirmationStatus.TENTATIVE, frame_id=10
    )
    wrong_class = make_object(
        "C-1",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=10,
        class_name="hammer",
    )
    still_tentative = make_object(
        "T-2", status=ConfirmationStatus.TENTATIVE, frame_id=10
    )

    decision = PartialDuplicateEvaluator().evaluate(
        tentative, [wrong_class, still_tentative], set()
    )

    assert decision.decision == "no_match"
    assert decision.reason == "no_same_class_confirmed"


def test_scale_and_distance_guards_report_structured_reasons():
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )
    full_scale = make_object(
        "T-scale",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        centroid=(50.0, 50.0),
        area=8_000.0,
    )
    far_fragment = make_object(
        "T-far",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(0.0, 0.0, 40.0, 40.0),
        centroid=(500.0, 500.0),
        area=1_600.0,
    )

    scale_decision = PartialDuplicateEvaluator().evaluate(full_scale, [confirmed], set())
    distance_decision = PartialDuplicateEvaluator().evaluate(
        far_fragment, [confirmed], set()
    )

    assert scale_decision.reason == "not_partial_scale"
    assert distance_decision.reason == "distance_exceeded"


def test_scale_uses_midpoint_while_distance_uses_confirmed_max_area():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(0.0, 0.0, 10.0, 10.0),
        centroid=(14.0, 0.0),
        area=100.0,
        area_range=(50.0, 250.0),
    )
    confirmed = make_object(
        "C-1",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=10,
        bbox_pixels=(0.0, 0.0, 20.0, 20.0),
        centroid=(0.0, 0.0),
        area=400.0,
        area_range=(100.0, 400.0),
    )

    decision = PartialDuplicateEvaluator().evaluate(tentative, [confirmed], set())

    assert decision.decision == "attributed"
    assert decision.normalized_distance == 0.7


def test_evaluation_does_not_mutate_inputs():
    tentative = make_object(
        "T-1",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=10,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    confirmed = make_object(
        "C-1", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )
    candidates = [confirmed]
    pairs = {frozenset(("T-1", "unrelated"))}
    before = pickle.dumps((tentative, candidates, pairs))

    PartialDuplicateEvaluator().evaluate(tentative, candidates, pairs)

    assert pickle.dumps((tentative, candidates, pairs)) == before
