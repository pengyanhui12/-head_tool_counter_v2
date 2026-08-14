"""报告一致性测试 — JSON/CSV/控制台口径统一，persistent ID 只分配给 reportable"""
import numpy as np
import pytest

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
    """R1: total_objects = confirmed + tentative + uncertain"""
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
    assert [obj["provisional_id"] for obj in report["objects"]] == ["P-0001", "P-0003"]


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
