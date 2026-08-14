"""合并策略测试 — co-occurrence 阻止合并，frame overlap 阻止合并，审计记录"""
import numpy as np
import pytest

from core.types import (
    GlobalDetection, GlobalObject, ConfirmationStatus,
    VisibilityStatus, ReviewFlag,
)
from core.object_associator import ObjectAssociator
from core.merge_policy import MergePolicy


def make_gd(frame_id=1, keyframe_id=1, track_id=0, class_name="wrench",
            centroid=(100.0, 200.0), area=5000.0, confidence=0.8,
            sharpness=80.0, mapping_quality=0.7, bbox_pixels=(0, 0, 100, 50)):
    return GlobalDetection(
        frame_id=frame_id,
        keyframe_id=keyframe_id,
        track_id=track_id,
        projected_corners=np.array([[0, 0], [100, 0], [100, 50], [0, 50]]),
        projected_center=(50.0, 25.0),
        polygon_centroid=centroid,
        polygon_area=area,
        class_id=0,
        class_name=class_name,
        detection_confidence=confidence,
        sharpness=sharpness,
        mapping_quality=mapping_quality,
        edge_quality=1.0,
        size_quality=1.0,
        transform_version=1,
        source="L1",
        bbox_pixels=bbox_pixels,
    )


def make_review_object(
    object_id,
    *,
    status,
    frame_id=10,
    bbox_pixels=(0.0, 0.0, 100.0, 100.0),
    centroid=(50.0, 50.0),
    area=10_000.0,
    observation_count=1,
    keyframe_ids=None,
    mapping_quality=0.7,
):
    observations = [
        make_gd(
            frame_id=frame_id + offset,
            keyframe_id=frame_id + offset,
            track_id=None,
            centroid=centroid,
            area=area,
            bbox_pixels=bbox_pixels,
            mapping_quality=mapping_quality,
        )
        for offset in range(observation_count)
    ]
    return GlobalObject(
        provisional_id=object_id,
        persistent_id=None,
        class_name="wrench",
        confirmation_status=status,
        visibility_status=VisibilityStatus.ACTIVE,
        observations=observations,
        centroid_xy=centroid,
        area_range=(area, area),
        keyframe_ids=set(keyframe_ids or {frame_id}),
        observation_count=observation_count,
    )


def add_review_objects(assoc, *objects):
    assoc.map._objects.extend(objects)


def test_merge_blocked_by_cooccurrence():
    """同帧共现对象永久禁止自动合并"""
    assoc = ObjectAssociator(debug_mode=False)
    # 创建两个同帧出现的对象（IoU=0 的 bbox 确保记录为共现）
    gd1 = make_gd(frame_id=1, track_id=1, centroid=(100, 100),
                  bbox_pixels=(0, 0, 50, 50))
    gd2 = make_gd(frame_id=1, track_id=2, centroid=(500, 500),
                  bbox_pixels=(60, 0, 110, 50))
    assoc.ingest_frame(1, [gd1, gd2])
    objs = assoc.map.get_all()
    assert len(objs) == 2, f"Expected 2 objects but got {len(objs)}"
    pair = frozenset([objs[0].provisional_id, objs[1].provisional_id])

    # 由于两个 bbox IoU=0 且同类，应该记录为共现
    # 但检测器也可能因为 class_id 不同等原因未记录
    # 我们手动设置共现状态以测试 merge 阻止逻辑
    assert pair in assoc._co_occurred_pairs

    # 标记为 CONFIRMED
    for o in objs:
        o.confirmation_status = ConfirmationStatus.CONFIRMED

    # 尝试合并 → 应被阻止
    assoc.final_review()
    # 两个对象应该仍然独立
    assert len(assoc.get_reportable_objects()) == 2


def test_cooccurrence_is_recorded_when_all_detections_match_existing_tracks():
    assoc = ObjectAssociator(debug_mode=False)
    assoc.ingest_frame(1, [
        make_gd(frame_id=1, track_id=1, centroid=(100, 100),
                bbox_pixels=(0, 0, 50, 50)),
        make_gd(frame_id=1, track_id=2, centroid=(500, 500),
                bbox_pixels=(60, 0, 110, 50)),
    ])
    assoc._co_occurred_pairs.clear()

    assoc.ingest_frame(2, [
        make_gd(frame_id=2, track_id=1, centroid=(101, 100),
                bbox_pixels=(0, 0, 50, 50)),
        make_gd(frame_id=2, track_id=2, centroid=(499, 500),
                bbox_pixels=(60, 0, 110, 50)),
    ])

    objs = assoc.map.get_all()
    pair = frozenset([objs[0].provisional_id, objs[1].provisional_id])
    assert pair in assoc._co_occurred_pairs


def test_overlapping_boxes_assigned_to_distinct_objects_are_cooccurrence():
    assoc = ObjectAssociator(debug_mode=False)
    assoc.ingest_frame(1, [
        make_gd(frame_id=1, track_id=1, centroid=(100, 100),
                bbox_pixels=(0, 0, 100, 100)),
        make_gd(frame_id=1, track_id=2, centroid=(120, 100),
                bbox_pixels=(70, 0, 170, 100)),
    ])

    objs = assoc.map.get_all()
    pair = frozenset([objs[0].provisional_id, objs[1].provisional_id])
    assert pair in assoc._co_occurred_pairs


def test_nested_same_frame_boxes_are_not_independent_cooccurrence():
    assoc = ObjectAssociator()
    assoc.ingest_frame(
        1,
        [
            make_gd(
                frame_id=1,
                track_id=1,
                centroid=(50.0, 50.0),
                area=10_000.0,
                bbox_pixels=(0.0, 0.0, 100.0, 100.0),
            ),
            make_gd(
                frame_id=1,
                track_id=2,
                centroid=(40.0, 40.0),
                area=1_600.0,
                bbox_pixels=(20.0, 20.0, 60.0, 60.0),
            ),
        ],
    )
    confirmed, tentative = assoc.map.get_all()
    confirmed.confirmation_status = ConfirmationStatus.CONFIRMED
    pair = frozenset((confirmed.provisional_id, tentative.provisional_id))

    assert pair in assoc._co_occurred_pairs
    assert pair not in assoc._independent_co_occurred_pairs

    assoc.final_review()

    assert tentative.likely_partial_duplicate_of == confirmed.provisional_id
    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE in tentative.review_flags


@pytest.mark.parametrize(
    ("second_box", "second_centroid"),
    [
        ((110.0, 0.0, 210.0, 100.0), (160.0, 50.0)),
        ((90.0, 0.0, 190.0, 100.0), (140.0, 50.0)),
    ],
    ids=["disjoint", "low-overlap"],
)
def test_separable_same_frame_boxes_are_independent_cooccurrence(
    second_box, second_centroid
):
    assoc = ObjectAssociator(
        independent_co_occurrence_max_containment=0.25
    )
    assoc.ingest_frame(
        1,
        [
            make_gd(
                frame_id=1,
                track_id=1,
                centroid=(50.0, 50.0),
                bbox_pixels=(0.0, 0.0, 100.0, 100.0),
            ),
            make_gd(
                frame_id=1,
                track_id=2,
                centroid=second_centroid,
                bbox_pixels=second_box,
            ),
        ],
    )
    first, second = assoc.map.get_all()
    pair = frozenset((first.provisional_id, second.provisional_id))

    assert pair in assoc._co_occurred_pairs
    assert pair in assoc._independent_co_occurred_pairs


@pytest.mark.parametrize(
    "invalid_box",
    [
        (0.0, 0.0, 0.0, 100.0),
        (0.0, 0.0, float("nan"), 100.0),
    ],
    ids=["zero-area", "non-finite"],
)
def test_invalid_raw_boxes_are_not_independent_cooccurrence(invalid_box):
    assoc = ObjectAssociator()
    assoc.ingest_frame(
        1,
        [
            make_gd(frame_id=1, track_id=1, bbox_pixels=invalid_box),
            make_gd(
                frame_id=1,
                track_id=2,
                centroid=(500.0, 500.0),
                bbox_pixels=(200.0, 0.0, 300.0, 100.0),
            ),
        ],
    )
    first, second = assoc.map.get_all()
    pair = frozenset((first.provisional_id, second.provisional_id))

    assert pair in assoc._co_occurred_pairs
    assert pair not in assoc._independent_co_occurred_pairs


def test_merge_blocked_by_frame_overlap():
    """observation frame 有重叠的对象不能合并"""
    policy = MergePolicy()
    primary = GlobalObject(
        provisional_id="P-0001", persistent_id=None, class_name="wrench",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
    )
    secondary = GlobalObject(
        provisional_id="P-0002", persistent_id=None, class_name="wrench",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
    )

    # 相同 frame
    gd = make_gd(frame_id=5, track_id=1)
    primary.observations.append(gd)
    primary.observation_count = 1
    secondary.observations.append(gd)
    secondary.observation_count = 1

    allowed, reason, audit = policy.can_merge(primary, secondary)
    assert not allowed
    assert reason == "observation_frame_overlap"


def test_merge_allowed_with_shared_track():
    """shared track + 全部安全条件 → 允许合并"""
    policy = MergePolicy()
    primary = GlobalObject(
        provisional_id="P-0001", persistent_id=None, class_name="wrench",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
    )
    secondary = GlobalObject(
        provisional_id="P-0002", persistent_id=None, class_name="wrench",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
    )

    # 不同 frame
    gd1 = make_gd(frame_id=1, track_id=1)
    gd2 = make_gd(frame_id=2, track_id=2)
    primary.observations.append(gd1)
    primary.observation_count = 1
    secondary.observations.append(gd2)
    secondary.observation_count = 1

    allowed, reason, audit = policy.can_merge(primary, secondary, shared_track_keys=[1])
    assert allowed
    assert audit.decision == "merged"
    assert audit.position_distance is not None


def test_merge_marks_likely_duplicate():
    """centroid < 30px 仅标记 LIKELY_DUPLICATE，不自动合并"""
    assoc = ObjectAssociator(debug_mode=False,
                              max_position_distance_px=500.0,
                              min_observations_confirmed=1,
                              min_keyframes_confirmed=1,
                              online_gate_ratio=0.6)
    # 创建两个不同 track、不同 frame、空间分离的对象（距离>300=500*0.6）
    gd1 = make_gd(frame_id=1, track_id=1, centroid=(100, 100))
    assoc.ingest_frame(1, [gd1])
    gd2 = make_gd(frame_id=5, track_id=2, centroid=(500, 500))
    assoc.ingest_frame(5, [gd2])

    objs = assoc.map.get_all()
    assert len(objs) == 2, f"Expected 2 objects but got {len(objs)}"
    for o in objs:
        o.confirmation_status = ConfirmationStatus.CONFIRMED

    # 手动设置 centroid 逼近来测试 close-duplicate 标记
    objs[0].centroid_xy = (100.0, 100.0)
    objs[1].centroid_xy = (115.0, 115.0)

    assoc.final_review()
    # 两个不同 track 的对象应该仍然独立（centroid close 不自动合并）
    # shared_track is empty so no merge happens
    reportable = assoc.get_reportable_objects()
    assert len(reportable) == 2, f"Expected 2 reportable but got {len(reportable)}"
    # 应该标记了 LIKELY_DUPLICATE（因为 centroid < 30px）
    has_duplicate_flag = any(
        ReviewFlag.LIKELY_DUPLICATE in o.review_flags for o in reportable
    )
    assert has_duplicate_flag, "Close centroids should mark LIKELY_DUPLICATE"


def test_merge_produces_audit_and_rejected_reason():
    """合并产生审计记录 + O6: REJECTED 有 rejected_reason"""
    assoc = ObjectAssociator(debug_mode=False)
    gd1 = make_gd(frame_id=1, track_id=1, centroid=(100, 100))
    assoc.ingest_frame(1, [gd1])
    gd2 = make_gd(frame_id=5, track_id=1, centroid=(105, 105))
    assoc.ingest_frame(5, [gd2])

    objs = assoc.map.get_all()
    for o in objs:
        o.confirmation_status = ConfirmationStatus.CONFIRMED

    assoc.final_review()
    # shared track 合并 → 至少一个 REJECTED
    rejected = [o for o in assoc.map.get_all()
                if o.confirmation_status == ConfirmationStatus.REJECTED]
    for r in rejected:
        assert r.rejected_reason is not None, f"O6: {r.provisional_id} has no rejected_reason"
        if "merged" in r.rejected_reason:
            assert r.merged_into_id is not None, f"O7: {r.provisional_id} has no merged_into_id"


def test_class_mismatch_blocks_merge():
    """不同类别对象不能合并"""
    policy = MergePolicy()
    primary = GlobalObject(
        provisional_id="P-0001", persistent_id=None, class_name="wrench",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
    )
    secondary = GlobalObject(
        provisional_id="P-0002", persistent_id=None, class_name="hammer",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
    )
    allowed, reason, audit = policy.can_merge(primary, secondary)
    assert not allowed
    assert reason == "class_mismatch"


def test_track_conflict_blocks_merge():
    """track conflict 对象不能合并"""
    policy = MergePolicy()
    primary = GlobalObject(
        provisional_id="P-0001", persistent_id=None, class_name="wrench",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
        review_flags={ReviewFlag.TRACK_CONFLICT},
    )
    secondary = GlobalObject(
        provisional_id="P-0002", persistent_id=None, class_name="wrench",
        confirmation_status=ConfirmationStatus.CONFIRMED,
        visibility_status=VisibilityStatus.ACTIVE,
    )
    allowed, reason, audit = policy.can_merge(primary, secondary)
    assert not allowed
    assert "track_conflict" in reason


def test_online_gate_uses_ratio_and_class_overrides():
    assoc = ObjectAssociator(
        max_position_distance_px=200.0,
        online_gate_ratio=0.5,
        per_class_gate_ratios={"pliers": 0.4},
        per_class_position_gates={"screwdriver": 60.0},
    )

    assert assoc._online_gate_for_class("hammer") == 100.0
    assert assoc._online_gate_for_class("pliers") == 80.0
    assert assoc._online_gate_for_class("screwdriver") == 60.0


def test_object_associator_preserves_positional_debug_mode_compatibility():
    legacy_positional_args = (
        120.0,
        0.55,
        0.20,
        0.10,
        0.15,
        0.75,
        3,
        2,
        0.60,
        3,
        None,
        0.50,
        None,
        None,
        15,
        30.0,
        True,
    )

    assoc = ObjectAssociator(*legacy_positional_args)

    assert assoc._debug_mode is True


def test_final_review_runs_merge_close_marking_then_partial_duplicate_review():
    assoc = ObjectAssociator()
    calls = []
    assoc._merge_by_shared_track_safe = lambda: calls.append("shared_track")
    assoc._mark_close_duplicates = lambda: calls.append("close_duplicates")
    assoc._review_tentative_partial_duplicates = lambda: calls.append(
        "partial_duplicates"
    )

    assoc.final_review()

    assert calls == ["shared_track", "close_duplicates", "partial_duplicates"]


def test_final_review_attributes_contained_tentative_without_mutating_objects():
    assoc = ObjectAssociator()
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED
    )
    confirmed.persistent_id = "GO-9999"
    tentative = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    add_review_objects(assoc, confirmed, tentative)
    original_confirmed_observations = tuple(confirmed.observations)
    original_observations = tuple(tentative.observations)

    assoc.final_review()

    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE in tentative.review_flags
    assert tentative.likely_partial_duplicate_of == confirmed.provisional_id
    assert tentative.duplicate_candidate_ids == []
    assert tentative.duplicate_evidence == {
        "decision": "likely_partial_duplicate",
        "containment_score": 1.0,
        "normalized_distance": pytest.approx(np.sqrt(200.0) / 100.0),
        "mapping_quality": 0.7,
        "co_occurrence_blocked": False,
        "reason": "unique_candidate",
        "candidates": [
            {
                "candidate_id": confirmed.provisional_id,
                "score": pytest.approx(
                    1.0 - np.sqrt(200.0) / 100.0
                ),
                "containment_score": 1.0,
                "normalized_distance": pytest.approx(
                    np.sqrt(200.0) / 100.0
                ),
                "mapping_quality": 0.7,
            }
        ],
    }
    assert tentative.confirmation_status == ConfirmationStatus.TENTATIVE
    assert confirmed.confirmation_status == ConfirmationStatus.CONFIRMED
    assert tuple(confirmed.observations) == original_confirmed_observations
    assert tuple(tentative.observations) == original_observations
    assert confirmed.observation_count == 1
    assert tentative.observation_count == 1
    assert len(assoc.map.get_all()) == 2


@pytest.mark.parametrize("mapping_quality", [0.1, float("nan")])
def test_final_review_does_not_flag_same_frame_candidate_with_unreliable_mapping(
    mapping_quality,
):
    assoc = ObjectAssociator()
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED
    )
    tentative = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
        mapping_quality=mapping_quality,
    )
    add_review_objects(assoc, confirmed, tentative)

    assoc.final_review()

    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE not in tentative.review_flags
    assert ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE not in tentative.review_flags
    assert tentative.likely_partial_duplicate_of is None
    assert tentative.duplicate_candidate_ids == []
    assert tentative.duplicate_evidence["decision"] == "no_match"
    assert tentative.duplicate_evidence["reason"] == "low_mapping_quality"
    assert tentative.duplicate_evidence["candidates"] == []


def test_associator_reportable_api_uses_four_status_counted_semantics():
    assoc = ObjectAssociator()
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED
    )
    tentative = make_review_object(
        "P-0002", status=ConfirmationStatus.TENTATIVE
    )
    uncertain = make_review_object(
        "P-0003", status=ConfirmationStatus.UNCERTAIN
    )
    rejected = make_review_object(
        "P-0004", status=ConfirmationStatus.REJECTED
    )
    add_review_objects(assoc, confirmed, tentative, uncertain, rejected)

    assert assoc.get_reportable_objects() == [confirmed, uncertain]


def test_final_review_independent_cooccurrence_blocks_partial_duplicate_attribution():
    assoc = ObjectAssociator()
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED
    )
    tentative = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    add_review_objects(assoc, confirmed, tentative)
    assoc._independent_co_occurred_pairs.add(
        frozenset((confirmed.provisional_id, tentative.provisional_id))
    )
    tentative.review_flags.add(ReviewFlag.LIKELY_PARTIAL_DUPLICATE)
    tentative.likely_partial_duplicate_of = confirmed.provisional_id
    tentative.duplicate_candidate_ids = [confirmed.provisional_id]
    tentative.duplicate_evidence = {"reason": "stale"}

    assoc.final_review()

    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE not in tentative.review_flags
    assert ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE not in tentative.review_flags
    assert tentative.likely_partial_duplicate_of is None
    assert tentative.duplicate_candidate_ids == []
    assert tentative.duplicate_evidence == {
        "decision": "no_match",
        "containment_score": None,
        "normalized_distance": None,
        "mapping_quality": None,
        "co_occurrence_blocked": True,
        "reason": "independent_co_occurrence",
        "candidates": [],
    }


def test_final_review_does_not_mark_complete_adjacent_objects_by_distance():
    assoc = ObjectAssociator(
        min_observations_confirmed=5,
        min_keyframes_confirmed=3,
    )
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED
    )
    complete_tentative = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        centroid=(55.0, 50.0),
        observation_count=5,
        keyframe_ids={10, 11, 12},
    )
    add_review_objects(assoc, confirmed, complete_tentative)
    complete_tentative.review_flags.add(ReviewFlag.LIKELY_PARTIAL_DUPLICATE)
    complete_tentative.likely_partial_duplicate_of = confirmed.provisional_id

    assoc.final_review()

    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE not in complete_tentative.review_flags
    assert ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE not in complete_tentative.review_flags
    assert complete_tentative.likely_partial_duplicate_of is None


def test_final_review_records_ambiguous_candidate_ids_without_attribution():
    assoc = ObjectAssociator()
    confirmed_a = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED, centroid=(40.0, 40.0)
    )
    confirmed_b = make_review_object(
        "P-0002", status=ConfirmationStatus.CONFIRMED, centroid=(45.0, 40.0)
    )
    tentative = make_review_object(
        "P-0003",
        status=ConfirmationStatus.TENTATIVE,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    add_review_objects(assoc, confirmed_b, tentative, confirmed_a)

    assoc.final_review()

    assert ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE in tentative.review_flags
    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE not in tentative.review_flags
    assert tentative.likely_partial_duplicate_of is None
    assert tentative.duplicate_candidate_ids == ["P-0001", "P-0002"]
    assert tentative.duplicate_evidence == {
        "decision": "ambiguous",
        "containment_score": 1.0,
        "normalized_distance": 0.0,
        "mapping_quality": 0.7,
        "co_occurrence_blocked": False,
        "reason": "candidate_margin_below_threshold",
        "candidates": [
            {
                "candidate_id": "P-0001",
                "score": 1.0,
                "containment_score": 1.0,
                "normalized_distance": 0.0,
                "mapping_quality": 0.7,
            },
            {
                "candidate_id": "P-0002",
                "score": 0.95,
                "containment_score": 1.0,
                "normalized_distance": 0.05,
                "mapping_quality": 0.7,
            },
        ],
    }


def test_repeated_final_review_is_idempotent_for_partial_duplicate_advice():
    assoc = ObjectAssociator()
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED
    )
    tentative = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    add_review_objects(assoc, confirmed, tentative)

    assoc.final_review()
    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE in tentative.review_flags
    first_state = (
        set(tentative.review_flags),
        tentative.likely_partial_duplicate_of,
        list(tentative.duplicate_candidate_ids),
        dict(tentative.duplicate_evidence),
        tentative.confirmation_status,
        tentative.observation_count,
        tuple(tentative.observations),
    )
    assoc.final_review()

    assert (
        set(tentative.review_flags),
        tentative.likely_partial_duplicate_of,
        list(tentative.duplicate_candidate_ids),
        dict(tentative.duplicate_evidence),
        tentative.confirmation_status,
        tentative.observation_count,
        tuple(tentative.observations),
    ) == first_state


def test_final_review_clears_stale_partial_duplicate_advice_on_no_match():
    assoc = ObjectAssociator()
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED
    )
    tentative = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    tentative.review_flags.add(ReviewFlag.LOW_CONFIDENCE)
    add_review_objects(assoc, confirmed, tentative)
    assoc.final_review()
    assert tentative.likely_partial_duplicate_of == confirmed.provisional_id

    tentative.review_flags.add(ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE)
    tentative.duplicate_candidate_ids = ["stale-id"]
    assoc._independent_co_occurred_pairs.add(
        frozenset((confirmed.provisional_id, tentative.provisional_id))
    )
    assoc.final_review()

    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE not in tentative.review_flags
    assert ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE not in tentative.review_flags
    assert ReviewFlag.LOW_CONFIDENCE in tentative.review_flags
    assert tentative.likely_partial_duplicate_of is None
    assert tentative.duplicate_candidate_ids == []
    assert tentative.duplicate_evidence["decision"] == "no_match"
    assert tentative.duplicate_evidence["reason"] == (
        "independent_co_occurrence"
    )
    assert tentative.duplicate_evidence["co_occurrence_blocked"] is True


def test_final_review_clears_partial_duplicate_advice_after_confirmation():
    assoc = ObjectAssociator()
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED
    )
    tentative = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    tentative.review_flags.add(ReviewFlag.LOW_CONFIDENCE)
    add_review_objects(assoc, confirmed, tentative)
    assoc.final_review()
    assert tentative.likely_partial_duplicate_of == confirmed.provisional_id

    tentative.confirmation_status = ConfirmationStatus.CONFIRMED
    assoc.final_review()

    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE not in tentative.review_flags
    assert ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE not in tentative.review_flags
    assert ReviewFlag.LOW_CONFIDENCE in tentative.review_flags
    assert tentative.likely_partial_duplicate_of is None
    assert tentative.duplicate_candidate_ids == []
    assert tentative.duplicate_evidence == {}


def test_final_review_clears_advice_when_shared_track_merge_rejects_tentative():
    assoc = ObjectAssociator()
    confirmed = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )
    tentative = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=11,
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    add_review_objects(assoc, confirmed, tentative)
    assoc.final_review()
    assert tentative.likely_partial_duplicate_of == confirmed.provisional_id

    confirmed.track_ids.add(7)
    tentative.track_ids.add(7)
    assoc.final_review()

    assert tentative.confirmation_status == ConfirmationStatus.REJECTED
    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE not in tentative.review_flags
    assert ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE not in tentative.review_flags
    assert tentative.likely_partial_duplicate_of is None
    assert tentative.duplicate_candidate_ids == []
    assert tentative.duplicate_evidence == {}


def test_final_review_preserves_merge_safety_lineage_without_advisory_block():
    assoc = ObjectAssociator()
    primary = make_review_object(
        "P-0001",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=10,
        observation_count=2,
        keyframe_ids={10, 11},
    )
    secondary = make_review_object(
        "P-0002",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=20,
    )
    tentative = make_review_object(
        "P-0003",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=20,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    primary.track_ids.add(7)
    secondary.track_ids.add(7)
    add_review_objects(assoc, primary, secondary, tentative)

    original_pair = frozenset(
        (secondary.provisional_id, tentative.provisional_id)
    )
    resolved_pair = frozenset(
        (primary.provisional_id, tentative.provisional_id)
    )
    assoc._co_occurred_pairs.add(original_pair)
    assoc._frame_co_occurred.add(
        (20, secondary.provisional_id, tentative.provisional_id)
    )
    assoc._merge_policy.record_co_occurrence(
        20, secondary.provisional_id, tentative.provisional_id
    )

    assoc.final_review()

    assert secondary.confirmation_status == ConfirmationStatus.REJECTED
    assert secondary.merged_into_id == primary.provisional_id
    assert original_pair in assoc._co_occurred_pairs
    assert resolved_pair in assoc._co_occurred_pairs
    assert assoc._merge_policy.have_co_occurred(
        secondary.provisional_id, tentative.provisional_id
    )
    assert assoc._merge_policy.have_co_occurred(
        primary.provisional_id, tentative.provisional_id
    )
    assert (
        20,
        secondary.provisional_id,
        tentative.provisional_id,
    ) in assoc._frame_co_occurred
    assert any(
        frame_id == 20
        and frozenset((pid_a, pid_b)) == resolved_pair
        for frame_id, pid_a, pid_b in assoc._frame_co_occurred
    )
    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE in tentative.review_flags
    assert ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE not in tentative.review_flags
    assert tentative.likely_partial_duplicate_of == primary.provisional_id
    assert tentative.duplicate_candidate_ids == []
    assert tentative.duplicate_evidence["decision"] == (
        "likely_partial_duplicate"
    )


def test_final_review_preserves_independent_cooccurrence_lineage_across_merge():
    assoc = ObjectAssociator()
    primary = make_review_object(
        "P-0001",
        status=ConfirmationStatus.CONFIRMED,
        frame_id=10,
        observation_count=2,
        keyframe_ids={10, 11},
    )
    secondary = make_review_object(
        "P-0002", status=ConfirmationStatus.TENTATIVE, frame_id=20
    )
    other = make_review_object(
        "P-0003",
        status=ConfirmationStatus.TENTATIVE,
        frame_id=30,
        bbox_pixels=(20.0, 20.0, 60.0, 60.0),
        centroid=(40.0, 40.0),
        area=1_600.0,
    )
    primary.track_ids.add(7)
    secondary.track_ids.add(7)
    add_review_objects(assoc, primary, secondary, other)
    original_pair = frozenset(
        (secondary.provisional_id, other.provisional_id)
    )
    resolved_pair = frozenset((primary.provisional_id, other.provisional_id))
    assoc._independent_co_occurred_pairs.add(original_pair)
    assoc._frame_independent_co_occurred.add(
        (20, secondary.provisional_id, other.provisional_id)
    )

    assoc.final_review()

    assert secondary.confirmation_status == ConfirmationStatus.REJECTED
    assert original_pair in assoc._independent_co_occurred_pairs
    assert resolved_pair in assoc._independent_co_occurred_pairs
    assert any(
        frame_id == 20 and frozenset((pid_a, pid_b)) == resolved_pair
        for frame_id, pid_a, pid_b in assoc._frame_independent_co_occurred
    )
    assert ReviewFlag.LIKELY_PARTIAL_DUPLICATE not in other.review_flags
    assert other.duplicate_evidence["reason"] == "independent_co_occurrence"
    assert other.duplicate_evidence["co_occurrence_blocked"] is True


def test_repeated_final_review_clears_preassigned_id_on_merged_secondary():
    assoc = ObjectAssociator()
    primary = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )
    secondary = make_review_object(
        "P-0002", status=ConfirmationStatus.CONFIRMED, frame_id=20
    )
    primary.persistent_id = "GO-0001"
    secondary.persistent_id = "GO-0002"
    primary.track_ids.add(7)
    secondary.track_ids.add(7)
    add_review_objects(assoc, primary, secondary)

    assoc.final_review()
    assoc.final_review()

    assert secondary.confirmation_status == ConfirmationStatus.REJECTED
    assert secondary.persistent_id is None
    assert assoc.validate_object_map()["persistent_id_on_rejected"] == []


def test_frameless_cooccurrence_lineage_blocks_a_cascading_merge():
    assoc = ObjectAssociator()
    primary = make_review_object(
        "P-0001", status=ConfirmationStatus.CONFIRMED, frame_id=10
    )
    secondary = make_review_object(
        "P-0002", status=ConfirmationStatus.TENTATIVE, frame_id=20
    )
    other = make_review_object(
        "P-0003", status=ConfirmationStatus.TENTATIVE, frame_id=30
    )
    primary.track_ids.add(7)
    secondary.track_ids.add(7)
    other.track_ids.add(7)
    add_review_objects(assoc, primary, secondary, other)

    original_pair = frozenset(
        (secondary.provisional_id, other.provisional_id)
    )
    resolved_pair = frozenset(
        (primary.provisional_id, other.provisional_id)
    )
    assoc._co_occurred_pairs.add(original_pair)

    assoc._merge_objects(primary, secondary)

    assert original_pair in assoc._co_occurred_pairs
    assert resolved_pair in assoc._co_occurred_pairs
    assert assoc._merge_policy.have_co_occurred(
        primary.provisional_id, other.provisional_id
    )

    assoc.final_review()

    assert other.confirmation_status == ConfirmationStatus.TENTATIVE
    assert other.merged_into_id is None
