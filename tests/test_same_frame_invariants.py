"""同帧不变量测试 — 一个对象同帧最多一个 observation"""
import numpy as np
import pytest

from core.types import (
    GlobalDetection, GlobalObject, ConfirmationStatus,
    VisibilityStatus, ReviewFlag,
)
from core.object_associator import ObjectAssociator
from core.exceptions import SameFrameObservationError


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


def test_one_observation_per_frame():
    """O2: 一个对象在同一 frame_id 最多一个 observation"""
    assoc = ObjectAssociator(debug_mode=False)
    gd1 = make_gd(frame_id=1, track_id=1, centroid=(100, 100))
    assoc.ingest_frame(1, [gd1])
    obj = assoc.map.get_all()[0]
    assert obj.observation_count == 1
    assert len(obj.observations) == obj.observation_count


def test_same_frame_two_detections_different_objects():
    """O3: 同一帧两个不同检测 → 两个不同对象"""
    assoc = ObjectAssociator(debug_mode=False)
    gd1 = make_gd(frame_id=1, track_id=1, centroid=(100, 100))
    gd2 = make_gd(frame_id=1, track_id=2, centroid=(500, 500))
    affected = assoc.ingest_frame(1, [gd1, gd2])
    assert len(affected) == 2
    assert affected[0] != affected[1]


def test_observation_count_matches_list_length():
    """O8: observation_count == len(observations)"""
    assoc = ObjectAssociator(debug_mode=False)
    for i in range(5):
        gd = make_gd(frame_id=i, track_id=i, centroid=(100 + i, 100 + i))
        assoc.ingest_frame(i, [gd])
    for obj in assoc.map.get_all():
        assert obj.observation_count == len(obj.observations)


def test_frame_ids_unique():
    """O9: 每个对象 observation frame_id 唯一"""
    assoc = ObjectAssociator(debug_mode=False)
    # 同 track 多帧 → 同一对象
    for i in range(5):
        gd = make_gd(frame_id=i, track_id=0, centroid=(100 + i * 2, 100 + i * 2))
        assoc.ingest_frame(i, [gd])
    for obj in assoc.map.get_all():
        frame_ids = [obs.frame_id for obs in obj.observations]
        assert len(frame_ids) == len(set(frame_ids)), \
            f"Duplicate frame_ids in {obj.provisional_id}: {frame_ids}"


def test_has_observation_in_frame():
    """_has_observation_in_frame 正确检测"""
    assoc = ObjectAssociator(debug_mode=False)
    gd = make_gd(frame_id=10)
    obj = assoc.map.create_object(gd)
    assert assoc._has_observation_in_frame(obj, 10) is True
    assert assoc._has_observation_in_frame(obj, 11) is False


def test_assigned_detection_indices_unique():
    """同一帧内 assigned_detection_indices 去重"""
    assoc = ObjectAssociator(debug_mode=False)
    gd1 = make_gd(frame_id=1, track_id=1, centroid=(100, 100))
    gd2 = make_gd(frame_id=1, track_id=2, centroid=(500, 500))
    affected = assoc.ingest_frame(1, [gd1, gd2])
    # 每个 detection 最多分配给一个对象
    assert len(affected) <= 2
    all_objects = assoc.map.get_all()
    frame1_obs = []
    for obj in all_objects:
        for obs in obj.observations:
            if obs.frame_id == 1:
                frame1_obs.append((obj.provisional_id, obs.track_id))
    assert len(frame1_obs) == 2  # Two detections, two observations across all objects
