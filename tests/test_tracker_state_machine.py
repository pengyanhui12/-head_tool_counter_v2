"""Tracker 状态机测试 — active/inactive/lost 互斥，class_cost，frame-gap 推进"""
import numpy as np
import pytest

from core.types import DetectionCandidate, Track
from core.simple_tracker import SimpleDetectionTracker


def make_det(frame_id=1, bbox=(100, 100, 200, 200), class_id=0,
             class_name="wrench", confidence=0.8, source="L2",
             w=640, h=480):
    return DetectionCandidate(
        frame_id=frame_id, bbox=bbox, class_id=class_id,
        class_name=class_name, confidence=confidence,
        source=source, image_width=w, image_height=h,
    )


def test_active_inactive_lost_disjoint():
    """T3: active、inactive、lost 集合互斥。"""
    tracker = SimpleDetectionTracker(max_missed_detection_frames=2,
                                     lost_reactivation_frames=5)
    # track 0 created
    tracker.update([make_det(frame_id=1)], frame_id=1)
    assert len(tracker.get_active_tracks()) == 1
    assert len(tracker.get_inactive_tracks()) == 0

    # miss frames 2,3,4 → inactive (missed=3 > max_missed=2)
    tracker.update([], frame_id=2)
    tracker.update([], frame_id=3)
    tracker.update([], frame_id=4)
    assert len(tracker.get_active_tracks()) == 0
    assert len(tracker.get_inactive_tracks()) == 1

    # miss more: frame 5..9 → at frame 9, missed=8 > lost_reactivation=5 → lost
    tracker.update([], frame_id=5)
    tracker.update([], frame_id=6)
    tracker.update([], frame_id=7)
    tracker.update([], frame_id=8)
    tracker.update([], frame_id=9)
    assert len(tracker.get_active_tracks()) == 0
    assert len(tracker.get_inactive_tracks()) == 0  # Now lost


def test_class_cost():
    """fix: _class_cost 使用 track 对象而不是 track_id"""
    tracker = SimpleDetectionTracker()

    # Create a track via update
    tracker.update([make_det(frame_id=1, class_name="wrench", class_id=0)], frame_id=1)
    track = tracker._tracks[0]

    # Same class
    same_det = make_det(frame_id=2, class_name="wrench", class_id=0)
    assert tracker._class_cost(track, same_det) == 0.0

    # Compatible class
    tracker._class_compat = {"wrench": ["pliers"]}
    compat_det = make_det(frame_id=2, class_name="pliers", class_id=1)
    assert tracker._class_cost(track, compat_det) == 0.5

    # Incompatible class
    other_det = make_det(frame_id=2, class_name="hammer", class_id=2)
    assert tracker._class_cost(track, other_det) == float("inf")


def test_class_compatibility_is_symmetric():
    tracker = SimpleDetectionTracker(
        class_compatibility={"wrench": ["pliers"]},
    )
    tracker.update([
        make_det(frame_id=1, class_name="pliers", class_id=1)
    ], frame_id=1)

    wrench = make_det(frame_id=2, class_name="wrench", class_id=0)

    assert tracker._class_cost(tracker._tracks[0], wrench) == 0.5


def test_advance_frame():
    """advance_frame 推进时间但不增加 missed"""
    tracker = SimpleDetectionTracker()
    tracker.update([make_det(frame_id=1)], frame_id=1)
    t = tracker._tracks[0]
    assert t.last_update_frame_id == 1

    tracker.advance_frame(5)
    assert t.last_update_frame_id == 5
    assert t.missed_frames == 0  # 不增加 missed


def test_missed_frames_advance_by_real_frame_gap():
    tracker = SimpleDetectionTracker(
        max_missed_detection_frames=2,
        lost_reactivation_frames=10,
    )
    tracker.update([make_det(frame_id=1)], frame_id=1)

    tracker.update([], frame_id=5)

    track = tracker._tracks[0]
    assert track.missed_frames == 4
    assert track.state == "inactive"


def test_lost_track_releases_quality_drop_state():
    tracker = SimpleDetectionTracker(
        max_missed_detection_frames=1,
        lost_reactivation_frames=2,
    )
    tracker.update([make_det(frame_id=1)], frame_id=1)
    tracker._track_in_drop[0] = True

    tracker.update([], frame_id=4)

    assert tracker._tracks[0].state == "lost"
    assert 0 not in tracker._track_in_drop


def test_last_update_and_detection_frame():
    """track 记录 last_update_frame_id 和 last_detection_frame_id"""
    tracker = SimpleDetectionTracker()
    tracker.update([make_det(frame_id=10)], frame_id=10)
    t = tracker._tracks[0]
    assert t.last_update_frame_id == 10
    assert t.last_detection_frame_id == 10

    # miss → last_update advances, last_detection stays
    tracker.update([], frame_id=11)
    assert t.last_update_frame_id == 11
    assert t.last_detection_frame_id == 10


def test_track_id_unique_in_results():
    """T1+T2: 一次 update 中 track_id 和 detection index 唯一"""
    tracker = SimpleDetectionTracker()
    # 两个同类的 track 和一个检测
    tracker.update([make_det(frame_id=1, bbox=(100, 100, 200, 200))], frame_id=1)
    tracker.update([make_det(frame_id=2, bbox=(300, 100, 400, 200))], frame_id=2)

    # One detection should match at most one track
    results = tracker.update(
        [make_det(frame_id=3, bbox=(110, 110, 210, 210))], frame_id=3
    )
    assert len(results) <= 1


def test_generation_field():
    """track 有 generation 字段用于 logical_key"""
    tracker = SimpleDetectionTracker()
    tracker.update([make_det(frame_id=1)], frame_id=1)
    t = tracker._tracks[0]
    assert hasattr(t, 'generation')
    assert t.generation == 0


def test_update_rejects_frame_id_regression_once_tracking_has_started():
    tracker = SimpleDetectionTracker()
    tracker.update([make_det(frame_id=10)], frame_id=10)

    with pytest.raises(ValueError, match="cannot move backwards"):
        tracker.update([make_det(frame_id=9)], frame_id=9)

    assert tracker.current_frame_id == 10
    assert tracker.time_regressions == 1
