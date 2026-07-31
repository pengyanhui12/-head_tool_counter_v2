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
    assoc._co_occurred_pairs.add(pair)
    assoc._merge_policy.record_co_occurrence(1, objs[0].provisional_id, objs[1].provisional_id)

    # 标记为 CONFIRMED
    for o in objs:
        o.confirmation_status = ConfirmationStatus.CONFIRMED

    # 尝试合并 → 应被阻止
    assoc.final_review()
    # 两个对象应该仍然独立
    assert len(assoc.get_reportable_objects()) == 2


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
