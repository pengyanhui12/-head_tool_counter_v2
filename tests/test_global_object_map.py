"""GlobalObjectMap 单元测试"""
from core.global_object_map import GlobalObjectMap
from core.types import (
    GlobalDetection,
    ConfirmationStatus,
    VisibilityStatus,
    ReviewFlag,
)
import numpy as np


def _gd(class_name="wrench", centroid=(100.0, 100.0), track_id=0, kf_id=0):
    return GlobalDetection(
        frame_id=0, keyframe_id=kf_id, track_id=track_id,
        projected_corners=np.zeros((4,2)),
        projected_center=centroid, polygon_centroid=centroid,
        polygon_area=100.0, class_id=0, class_name=class_name,
        detection_confidence=0.9, sharpness=100.0, mapping_quality=0.8,
        edge_quality=1.0, size_quality=1.0, transform_version=0, source="L1",
    )


def test_create_and_retrieve():
    m = GlobalObjectMap()
    obj = m.create_object(_gd())
    assert obj.provisional_id == "P-0001"
    assert obj.confirmation_status == ConfirmationStatus.TENTATIVE
    assert obj.visibility_status == VisibilityStatus.ACTIVE
    assert m.get_by_provisional("P-0001") is obj


def test_assign_persistent_ids():
    m = GlobalObjectMap()
    confirmed = m.create_object(_gd())
    tentative = m.create_object(_gd())
    uncertain = m.create_object(_gd())
    rejected = m.create_object(_gd())
    m.set_confirmation(confirmed.provisional_id, ConfirmationStatus.CONFIRMED)
    m.set_confirmation(uncertain.provisional_id, ConfirmationStatus.UNCERTAIN)
    m.set_confirmation(rejected.provisional_id, ConfirmationStatus.REJECTED)

    m.assign_persistent_ids()

    assert confirmed.persistent_id == "GO-0001"
    assert tentative.persistent_id is None
    assert uncertain.persistent_id == "GO-0002"
    assert rejected.persistent_id is None


def test_set_confirmation_clears_persistent_id_when_demoted_to_tentative():
    m = GlobalObjectMap()
    obj = m.create_object(_gd())
    m.set_confirmation(obj.provisional_id, ConfirmationStatus.CONFIRMED)
    m.assign_persistent_ids()

    m.set_confirmation(obj.provisional_id, ConfirmationStatus.TENTATIVE)
    later_counted = m.create_object(_gd())
    m.set_confirmation(later_counted.provisional_id, ConfirmationStatus.CONFIRMED)
    m.assign_persistent_ids()

    assert obj.persistent_id is None
    assert later_counted.persistent_id == "GO-0002"


def test_set_confirmation_clears_persistent_id_when_demoted_to_rejected():
    m = GlobalObjectMap()
    obj = m.create_object(_gd())
    m.set_confirmation(obj.provisional_id, ConfirmationStatus.UNCERTAIN)
    m.assign_persistent_ids()

    m.set_confirmation(obj.provisional_id, ConfirmationStatus.REJECTED)
    later_counted = m.create_object(_gd())
    m.set_confirmation(later_counted.provisional_id, ConfirmationStatus.UNCERTAIN)
    m.assign_persistent_ids()

    assert obj.persistent_id is None
    assert later_counted.persistent_id == "GO-0002"


def test_get_reportable_returns_only_counted_objects():
    m = GlobalObjectMap()
    confirmed = m.create_object(_gd())
    tentative = m.create_object(_gd())
    uncertain = m.create_object(_gd())
    rejected = m.create_object(_gd())
    m.set_confirmation(confirmed.provisional_id, ConfirmationStatus.CONFIRMED)
    m.set_confirmation(uncertain.provisional_id, ConfirmationStatus.UNCERTAIN)
    m.set_confirmation(rejected.provisional_id, ConfirmationStatus.REJECTED)

    assert m.get_reportable() == [confirmed, uncertain]
    assert tentative not in m.get_reportable()
    assert rejected not in m.get_reportable()


def test_set_confirmation():
    m = GlobalObjectMap()
    obj = m.create_object(_gd())
    m.set_confirmation("P-0001", ConfirmationStatus.CONFIRMED)
    assert obj.confirmation_status == ConfirmationStatus.CONFIRMED


def test_set_visibility():
    m = GlobalObjectMap()
    obj = m.create_object(_gd())
    m.set_visibility("P-0001", VisibilityStatus.INACTIVE)
    assert obj.visibility_status == VisibilityStatus.INACTIVE
    # CONFIRMED + INACTIVE should coexist
    m.set_confirmation("P-0001", ConfirmationStatus.CONFIRMED)
    assert obj.confirmation_status == ConfirmationStatus.CONFIRMED
    assert obj.visibility_status == VisibilityStatus.INACTIVE


def test_add_review_flag():
    m = GlobalObjectMap()
    m.create_object(_gd())
    m.add_review_flag("P-0001", ReviewFlag.LIKELY_DUPLICATE)
    obj = m.get_by_provisional("P-0001")
    assert ReviewFlag.LIKELY_DUPLICATE in obj.review_flags


def test_summary():
    m = GlobalObjectMap()
    m.create_object(_gd(class_name="wrench"))
    m.create_object(_gd(class_name="plier"))
    m.set_confirmation("P-0001", ConfirmationStatus.CONFIRMED)
    s = m.summary
    assert s.get("confirmed", 0) == 1
    assert s.get("tentative", 0) == 1
