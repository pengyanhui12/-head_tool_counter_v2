"""track-object 绑定测试 — 禁止静默重绑定"""
import numpy as np
import pytest

from core.types import (
    GlobalDetection, GlobalObject, ConfirmationStatus,
    VisibilityStatus, ReviewFlag,
)
from core.object_associator import ObjectAssociator
from core.exceptions import TrackBindingConflict


def make_gd(frame_id=1, keyframe_id=1, track_id=0, class_name="wrench",
            centroid=(100.0, 200.0), area=5000.0, confidence=0.8,
            sharpness=80.0, mapping_quality=0.7):
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
        bbox_pixels=(0, 0, 100, 50),
    )


def test_first_binding():
    """首次绑定成功"""
    assoc = ObjectAssociator(debug_mode=True)
    gd = make_gd(track_id=1)
    assoc.ingest_frame(1, [gd])
    logical_key = assoc._make_logical_key(1)
    assert logical_key in assoc._track_to_object
    obj = assoc.map.get_by_provisional(assoc._track_to_object[logical_key])
    assert obj is not None
    assert obj.class_name == "wrench"


def test_same_object_repeated_binding():
    """同对象重复绑定不抛异常"""
    assoc = ObjectAssociator(debug_mode=True)
    gd = make_gd(track_id=1)
    assoc.ingest_frame(1, [gd])
    logical_key = assoc._make_logical_key(1)
    obj_id = assoc._track_to_object[logical_key]
    # 同对象不应失败
    assoc._bind_track_to_object(logical_key, obj_id)
    assert assoc._track_to_object[logical_key] == obj_id


def test_different_object_rebinding_raises():
    """不同对象重绑定在 debug_mode 抛异常"""
    assoc = ObjectAssociator(debug_mode=True)
    logical_key = assoc._make_logical_key(1)
    assoc._track_to_object[logical_key] = "P-0001"
    with pytest.raises(TrackBindingConflict):
        assoc._bind_track_to_object(logical_key, "P-0002")


def test_different_object_rebinding_non_debug_marks_conflict():
    """非 debug 模式：重绑定不抛异常，但标记 TRACK_CONFLICT"""
    assoc = ObjectAssociator(debug_mode=False)
    # 先创建对象
    gd = make_gd(track_id=1)
    obj = assoc.map.create_object(gd)
    logical_key = assoc._make_logical_key(1)
    assoc._track_to_object[logical_key] = obj.provisional_id

    # 另一个对象
    gd2 = make_gd(track_id=2, centroid=(200, 200))
    obj2 = assoc.map.create_object(gd2)

    # 尝试重绑定 → 不抛异常，标记冲突
    assoc._bind_track_to_object(logical_key, obj2.provisional_id)
    assert assoc.stats["track_binding_conflicts"] == 1
    assert ReviewFlag.TRACK_CONFLICT in obj.review_flags
    assert ReviewFlag.TRACK_CONFLICT in obj2.review_flags


def test_hungarian_match_binding():
    """匈牙利匹配后 track 绑定到对象"""
    assoc = ObjectAssociator(debug_mode=True)
    gd1 = make_gd(frame_id=1, track_id=10, centroid=(100, 100))
    affected = assoc.ingest_frame(1, [gd1])
    obj1_id = affected[0]

    # 同一 track 第2帧
    gd2 = make_gd(frame_id=2, track_id=10, centroid=(105, 105))
    affected = assoc.ingest_frame(2, [gd2])
    # 应该匹配到同一对象
    assert affected[0] == obj1_id
    obj = assoc.map.get_by_provisional(obj1_id)
    assert obj.observation_count == 2


def test_track_binding_conflict_does_not_update_wrong_object():
    """冲突不污染旧对象"""
    assoc = ObjectAssociator(debug_mode=False)
    # 创建对象和绑定
    gd1 = make_gd(frame_id=1, track_id=1, centroid=(100, 100))
    assoc.ingest_frame(1, [gd1])
    logical_key = assoc._make_logical_key(1)
    obj1_id = assoc._track_to_object[logical_key]
    obj1 = assoc.map.get_by_provisional(obj1_id)
    old_obs_count = obj1.observation_count

    # 尝试把 track_id=1 绑定到另一个新对象（这不是通过 ingest_frame 的正常路径）
    gd2 = make_gd(frame_id=2, track_id=2, centroid=(500, 500))
    obj2 = assoc.map.create_object(gd2)

    # 直接调用 _bind_track_to_object 测试冲突处理
    assoc._bind_track_to_object(logical_key, obj2.provisional_id)

    # 冲突计数增加了
    assert assoc.stats["track_binding_conflicts"] == 1
    # obj1 仍然绑定到旧 logical_key（未覆盖）
    assert assoc._track_to_object[logical_key] == obj1.provisional_id
    # obj1 没有被错误更新
    assert obj1.observation_count == old_obs_count
    # obj1 和 obj2 都标记了 TRACK_CONFLICT
    assert ReviewFlag.TRACK_CONFLICT in obj1.review_flags
    assert ReviewFlag.TRACK_CONFLICT in obj2.review_flags
