"""报告一致性测试 — JSON/CSV/控制台口径统一，persistent ID 只分配给 reportable"""
import csv
import io
import json

import numpy as np
import pytest

from core import report_generator
from core.types import (
    GlobalObject, GlobalDetection, ConfirmationStatus,
    VisibilityStatus, ReviewFlag,
)
from core.global_object_map import GlobalObjectMap
from core.report_generator import (
    ObjectPartitions,
    ReportGenerator,
    get_counted_objects,
    get_reportable_objects,
    get_review_candidates,
    partition_objects,
)


def make_obj(pid="P-0001", class_name="wrench", status=ConfirmationStatus.CONFIRMED):
    return GlobalObject(
        provisional_id=pid, persistent_id=None, class_name=class_name,
        confirmation_status=status,
        visibility_status=VisibilityStatus.ACTIVE,
    )


def make_gd(frame_id=1):
    return GlobalDetection(
        frame_id=frame_id, keyframe_id=frame_id, track_id=1,
        projected_corners=np.array([[0, 0], [100, 0], [100, 50], [0, 50]]),
        projected_center=(50, 25), polygon_centroid=(100, 200),
        polygon_area=5000, class_id=0, class_name="wrench",
        detection_confidence=0.8, sharpness=80.0,
        mapping_quality=0.7, edge_quality=1.0,
        size_quality=1.0, transform_version=1, source="L1",
        bbox_pixels=(0, 0, 100, 50),
    )


def test_partition_objects_separates_all_statuses_in_input_order():
    """R1: total_objects = confirmed + uncertain; tentative is review-only."""
    obj_map = GlobalObjectMap()
    o1 = make_obj("P-0001", "wrench", ConfirmationStatus.CONFIRMED)
    o2 = make_obj("P-0002", "wrench", ConfirmationStatus.TENTATIVE)
    o3 = make_obj("P-0003", "hammer", ConfirmationStatus.UNCERTAIN)
    o4 = make_obj("P-0004", "wrench", ConfirmationStatus.REJECTED)
    o4.rejected_reason = "test_rejection"

    obj_map._objects = [o1, o2, o3, o4]
    partitions = partition_objects(obj_map.get_all())

    assert isinstance(partitions, ObjectPartitions)
    assert partitions.counted == (o1, o3)
    assert partitions.review_candidates == (o2,)
    assert partitions.rejected == (o4,)


def test_counted_helpers_exclude_tentative_and_rejected():
    objects = [
        make_obj("P-0001", status=ConfirmationStatus.CONFIRMED),
        make_obj("P-0002", status=ConfirmationStatus.TENTATIVE),
        make_obj("P-0003", status=ConfirmationStatus.UNCERTAIN),
        make_obj("P-0004", status=ConfirmationStatus.REJECTED),
    ]

    counted = get_counted_objects(objects)

    assert counted == [objects[0], objects[2]]
    assert get_reportable_objects(objects) == counted
    assert get_review_candidates(objects) == [objects[1]]


def test_display_objects_are_canonical_counted_then_review_candidates():
    objects = [
        make_obj("P-0001", status=ConfirmationStatus.TENTATIVE),
        make_obj("P-0002", status=ConfirmationStatus.REJECTED),
        make_obj("P-0003", status=ConfirmationStatus.UNCERTAIN),
        make_obj("P-0004", status=ConfirmationStatus.CONFIRMED),
    ]

    assert report_generator.get_display_objects(objects) == [
        objects[2],
        objects[3],
        objects[0],
    ]


def test_r2_rejected_not_in_total():
    """R2: rejected 不进入 total_objects"""
    obj_map = GlobalObjectMap()
    o1 = make_obj("P-0001", "wrench", ConfirmationStatus.CONFIRMED)
    o2 = make_obj("P-0002", "wrench", ConfirmationStatus.TENTATIVE)
    o3 = make_obj("P-0003", "hammer", ConfirmationStatus.UNCERTAIN)
    o4 = make_obj("P-0004", "wrench", ConfirmationStatus.REJECTED)
    o4.rejected_reason = "test"
    obj_map._objects = [o1, o2, o3, o4]

    gen = ReportGenerator(object_map=obj_map)
    report = gen.generate_json_report()
    assert report["total_objects"] == report["confirmed_count"] + report["uncertain_count"]
    assert report["total_objects"] == 2
    assert report["tentative_count"] == 1
    assert report["rejected_count"] == 1
    assert [obj["provisional_id"] for obj in report["objects"]] == [
        "P-0001", "P-0003",
    ]
    assert [obj["provisional_id"] for obj in report["review_candidates"]] == [
        "P-0002",
    ]
    assert report["class_counts"] == {"wrench": 1, "hammer": 1}


def test_json_contract_separates_partitions_and_resolves_duplicate_ids():
    obj_map = GlobalObjectMap()
    confirmed = make_obj("P-0001", "wrench", ConfirmationStatus.CONFIRMED)
    confirmed.persistent_id = "GO-0001"
    confirmed.review_flags.add(ReviewFlag.LIKELY_PARTIAL_DUPLICATE)
    uncertain = make_obj("P-0002", "hammer", ConfirmationStatus.UNCERTAIN)
    uncertain.persistent_id = "GO-0002"
    uncertain.review_flags.add(ReviewFlag.LOW_CONFIDENCE)
    tentative = make_obj("P-0003", "plier", ConfirmationStatus.TENTATIVE)
    tentative.review_flags.update({
        ReviewFlag.LIKELY_PARTIAL_DUPLICATE,
        ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE,
    })
    tentative.likely_partial_duplicate_of = confirmed.provisional_id
    tentative.duplicate_candidate_ids = ["P-9999", uncertain.provisional_id,
                                         confirmed.provisional_id]
    tentative.duplicate_evidence = {"z": 1, "a": {"score": 0.75}}
    rejected = make_obj("P-0004", "saw", ConfirmationStatus.REJECTED)
    rejected.rejected_reason = "invalid"
    obj_map._objects = [confirmed, uncertain, tentative, rejected]

    report = ReportGenerator(object_map=obj_map).generate_json_report()

    assert report["schema_version"] == report_generator.REPORT_SCHEMA_VERSION
    assert report["total_objects"] == 2
    assert report["total_objects"] == (
        report["confirmed_count"] + report["uncertain_count"]
    )
    assert report["review_candidate_count"] == 1
    assert report["tentative_count"] == report["review_candidate_count"]
    assert report["rejected_count"] == 1
    assert report["review_required_count"] == 3
    assert report["likely_partial_duplicate_count"] == 1
    assert report["class_counts"] == {"wrench": 1, "hammer": 1}
    assert [item["provisional_id"] for item in report["objects"]] == [
        "P-0001", "P-0002",
    ]
    assert [item["provisional_id"] for item in report["review_candidates"]] == [
        "P-0003",
    ]
    assert [item["provisional_id"] for item in report["rejected_objects"]] == [
        "P-0004",
    ]

    review = report["review_candidates"][0]
    assert review["persistent_id"] is None
    assert review["counted"] is False
    assert review["likely_partial_duplicate_of"] == "GO-0001"
    assert review["duplicate_candidate_ids"] == ["GO-0001", "GO-0002", "P-9999"]
    assert review["duplicate_evidence"] == {"z": 1, "a": {"score": 0.75}}
    assert all(item["counted"] is True for item in report["objects"])


def test_json_and_csv_serialize_all_duplicate_decisions():
    obj_map = GlobalObjectMap()
    confirmed_a = make_obj(
        "P-0001", "wrench", ConfirmationStatus.CONFIRMED
    )
    confirmed_a.persistent_id = "GO-0001"
    confirmed_b = make_obj(
        "P-0002", "wrench", ConfirmationStatus.CONFIRMED
    )
    confirmed_b.persistent_id = "GO-0002"
    confirmed_c = make_obj(
        "P-0003", "wrench", ConfirmationStatus.CONFIRMED
    )
    confirmed_c.persistent_id = "GO-0003"

    likely = make_obj(
        "P-0004", "wrench", ConfirmationStatus.TENTATIVE
    )
    likely.review_flags.add(ReviewFlag.LIKELY_PARTIAL_DUPLICATE)
    likely.likely_partial_duplicate_of = confirmed_a.provisional_id
    likely.duplicate_evidence = {
        "decision": "likely_partial_duplicate",
        "containment_score": 0.9,
        "normalized_distance": 0.2,
        "mapping_quality": 0.8,
        "co_occurrence_blocked": False,
        "reason": "unique_candidate",
        "candidates": [
            {
                "candidate_id": confirmed_a.provisional_id,
                "score": 0.7,
                "containment_score": 0.9,
                "normalized_distance": 0.2,
                "mapping_quality": 0.8,
            }
        ],
    }

    ambiguous = make_obj(
        "P-0005", "wrench", ConfirmationStatus.TENTATIVE
    )
    ambiguous.review_flags.add(
        ReviewFlag.AMBIGUOUS_DUPLICATE_CANDIDATE
    )
    ambiguous.duplicate_candidate_ids = [
        confirmed_b.provisional_id,
        confirmed_c.provisional_id,
    ]
    ambiguous.duplicate_evidence = {
        "decision": "ambiguous",
        "containment_score": 0.9,
        "normalized_distance": 0.1,
        "mapping_quality": 0.8,
        "co_occurrence_blocked": False,
        "reason": "candidate_margin_below_threshold",
        "candidates": [
            {
                "candidate_id": confirmed_b.provisional_id,
                "score": 0.8,
                "containment_score": 0.9,
                "normalized_distance": 0.1,
                "mapping_quality": 0.8,
            },
            {
                "candidate_id": confirmed_c.provisional_id,
                "score": 0.75,
                "containment_score": 0.9,
                "normalized_distance": 0.15,
                "mapping_quality": 0.8,
            },
        ],
    }

    no_match = make_obj(
        "P-0006", "wrench", ConfirmationStatus.TENTATIVE
    )
    no_match.duplicate_evidence = {
        "decision": "no_match",
        "containment_score": None,
        "normalized_distance": None,
        "mapping_quality": None,
        "co_occurrence_blocked": True,
        "reason": "independent_co_occurrence",
        "candidates": [],
    }
    obj_map._objects = [
        confirmed_a,
        confirmed_b,
        confirmed_c,
        likely,
        ambiguous,
        no_match,
    ]

    generator = ReportGenerator(object_map=obj_map)
    report = generator.generate_json_report()
    reviews = {
        item["provisional_id"]: item
        for item in report["review_candidates"]
    }

    assert reviews["P-0004"]["duplicate_evidence"]["decision"] == (
        "likely_partial_duplicate"
    )
    assert reviews["P-0004"]["duplicate_evidence"]["candidates"][0][
        "candidate_id"
    ] == "GO-0001"
    assert reviews["P-0005"]["duplicate_evidence"]["decision"] == (
        "ambiguous"
    )
    assert [
        candidate["candidate_id"]
        for candidate in reviews["P-0005"]["duplicate_evidence"][
            "candidates"
        ]
    ] == ["GO-0002", "GO-0003"]
    assert reviews["P-0006"]["duplicate_evidence"]["decision"] == (
        "no_match"
    )
    assert reviews["P-0006"]["review_flags"] == []
    assert reviews["P-0006"]["likely_partial_duplicate_of"] is None

    rows = {
        row["provisional_id"]: row
        for row in csv.DictReader(
            io.StringIO(generator.generate_csv_report(obj_map.get_all()))
        )
    }
    assert rows["P-0004"]["review_status"] == "likely_partial_duplicate"
    assert rows["P-0005"]["review_status"] == "ambiguous"
    assert rows["P-0006"]["review_status"] == "no_match"
    for provisional_id in ("P-0004", "P-0005", "P-0006"):
        assert json.loads(rows[provisional_id]["duplicate_evidence"]) == (
            reviews[provisional_id]["duplicate_evidence"]
        )


def test_r4_persistent_id_only_for_reportable():
    """R4: persistent_id 只分配给最终 reportable objects"""
    obj_map = GlobalObjectMap()
    gd = make_gd(1)
    o1 = obj_map.create_object(gd)
    o1.confirmation_status = ConfirmationStatus.CONFIRMED
    o2 = obj_map.create_object(gd)
    o2.confirmation_status = ConfirmationStatus.REJECTED
    o2.rejected_reason = "test"

    obj_map.assign_persistent_ids()
    assert o1.persistent_id is not None
    assert o2.persistent_id is None  # REJECTED 不应有 persistent ID


def test_report_json_includes_rejected_audit():
    """JSON report 包含 rejected 对象审计信息"""
    obj_map = GlobalObjectMap()
    o1 = make_obj("P-0001", "wrench", ConfirmationStatus.CONFIRMED)
    o2 = make_obj("P-0002", "wrench", ConfirmationStatus.REJECTED)
    o2.rejected_reason = "merged_duplicate"
    o2.merged_into_id = "P-0001"
    o2.rejection_evidence = {"shared_tracks": [1]}
    obj_map._objects = [o1, o2]

    gen = ReportGenerator(object_map=obj_map)
    report = gen.generate_json_report()

    assert "rejected_objects" in report
    assert len(report["rejected_objects"]) == 1
    assert report["rejected_objects"][0]["rejected_reason"] == "merged_duplicate"
    assert report["rejected_objects"][0]["merged_into_id"] == "P-0001"


def test_csv_includes_rejected():
    """CSV 包含所有对象（含 REJECTED）"""
    obj_map = GlobalObjectMap()
    o1 = make_obj("P-0001", "wrench", ConfirmationStatus.CONFIRMED)
    o2 = make_obj("P-0002", "wrench", ConfirmationStatus.REJECTED)
    o2.rejected_reason = "merged_duplicate"
    obj_map._objects = [o1, o2]

    gen = ReportGenerator(object_map=obj_map)
    csv = gen.generate_csv_report(obj_map.get_all())
    lines = csv.strip().split("\n")
    assert len(lines) == 3  # header + 2 data rows
    assert "\r" not in csv


def test_csv_contract_includes_all_rows_and_deterministic_advisory_fields():
    obj_map = GlobalObjectMap()
    confirmed = make_obj("P-0001", "wrench", ConfirmationStatus.CONFIRMED)
    confirmed.persistent_id = "GO-0001"
    tentative = make_obj("P-0002", "wrench", ConfirmationStatus.TENTATIVE)
    tentative.review_flags.add(ReviewFlag.LIKELY_PARTIAL_DUPLICATE)
    tentative.likely_partial_duplicate_of = "P-0001"
    tentative.duplicate_candidate_ids = ["P-9999", "P-0001"]
    tentative.duplicate_evidence = {"z": 1, "a": 2}
    rejected = make_obj("P-0003", "hammer", ConfirmationStatus.REJECTED)
    obj_map._objects = [confirmed, tentative, rejected]

    csv_text = ReportGenerator(object_map=obj_map).generate_csv_report(
        obj_map.get_all()
    )
    rows = list(csv.DictReader(io.StringIO(csv_text)))

    assert len(rows) == 3
    assert [row["counted"] for row in rows] == ["true", "false", "false"]
    assert [row["review_status"] for row in rows] == [
        "", "likely_partial_duplicate", "rejected",
    ]
    assert rows[1]["likely_partial_duplicate_of"] == "GO-0001"
    assert rows[1]["duplicate_candidate_ids"] == "GO-0001|P-9999"
    assert rows[1]["duplicate_evidence"] == '{"a":2,"z":1}'
    assert "\r" not in csv_text


def test_observation_field_consistency():
    """报告字段: observation_frame_count, observation_frame_ids_unique"""
    obj_map = GlobalObjectMap()
    o = make_obj("P-0001", "wrench", ConfirmationStatus.CONFIRMED)
    gd1 = make_gd(frame_id=1)
    gd2 = make_gd(frame_id=5)
    o.observations = [gd1, gd2]
    o.observation_count = 2
    obj_map._objects = [o]

    gen = ReportGenerator(object_map=obj_map)
    report = gen.generate_json_report()
    obj_data = report["objects"][0]
    assert obj_data["observation_frame_count"] == 2
    assert obj_data["observation_frame_ids_unique"] is True
