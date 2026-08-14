"""SimpleDetectionTracker 单元测试"""
from core.simple_tracker import SimpleDetectionTracker
from core.types import DetectionCandidate


def _cand(frame_id: int, class_id: int, class_name: str,
           bbox: tuple, confidence: float = 0.9) -> DetectionCandidate:
    return DetectionCandidate(
        frame_id=frame_id, bbox=bbox, class_id=class_id,
        class_name=class_name, confidence=confidence,
        source="L1", image_width=640, image_height=480,
    )


def test_preview_no_detections():
    tracker = SimpleDetectionTracker()
    preview = tracker.preview([])
    assert not preview.l2_new_unmatched_detection
    assert not preview.track_quality_drop


def test_update_creates_tracks():
    tracker = SimpleDetectionTracker()
    dets = [_cand(0, 0, "wrench", (100, 100, 200, 200))]
    result = tracker.update(dets)
    assert len(result) == 1
    assert result[0].is_new_track
    assert result[0].track_id == 0


def test_update_matches_existing():
    tracker = SimpleDetectionTracker()
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))])
    result = tracker.update([_cand(1, 0, "wrench", (105, 105, 195, 195))])
    assert len(result) == 1
    assert result[0].track_id == 0  # same track
    assert not result[0].is_new_track


def test_unmatched_becomes_new_track():
    tracker = SimpleDetectionTracker()
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))])
    result = tracker.update([_cand(1, 1, "plier", (300, 300, 400, 400))])
    assert len(result) == 1
    assert result[0].is_new_track
    assert result[0].track_id == 1


def test_preview_detects_new_unmatched():
    tracker = SimpleDetectionTracker(new_detection_confirmation_runs=1)
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))])
    preview = tracker.preview([_cand(1, 1, "plier", (300, 300, 400, 400))])
    assert preview.l2_new_unmatched_detection


def test_preview_uses_one_to_one_assignment():
    tracker = SimpleDetectionTracker(new_detection_confirmation_runs=1)
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))])

    preview = tracker.preview([
        _cand(1, 0, "wrench", (100, 100, 200, 200)),
        _cand(1, 0, "wrench", (105, 105, 195, 195)),
    ])

    assert len(preview.unmatched_detection_indices) == 1
    assert preview.l2_new_unmatched_detection


def test_preview_requires_three_consecutive_matches_for_new_detection():
    tracker = SimpleDetectionTracker(new_detection_confirmation_runs=3)

    first = tracker.preview([_cand(1, 1, "plier", (300, 300, 400, 400))])
    second = tracker.preview([_cand(2, 1, "plier", (302, 301, 402, 401))])
    third = tracker.preview([_cand(3, 1, "plier", (304, 302, 404, 402))])
    fourth = tracker.preview([_cand(4, 1, "plier", (306, 303, 406, 403))])

    assert not first.l2_new_unmatched_detection
    assert not second.l2_new_unmatched_detection
    assert third.l2_new_unmatched_detection
    assert not fourth.l2_new_unmatched_detection


def test_preview_does_not_accumulate_different_unmatched_objects():
    tracker = SimpleDetectionTracker(new_detection_confirmation_runs=3)

    assert not tracker.preview([
        _cand(1, 1, "plier", (50, 50, 100, 100))
    ]).l2_new_unmatched_detection
    assert not tracker.preview([
        _cand(2, 1, "plier", (300, 300, 350, 350))
    ]).l2_new_unmatched_detection
    assert not tracker.preview([
        _cand(3, 1, "plier", (50, 50, 100, 100))
    ]).l2_new_unmatched_detection


def test_preview_empty_l2_run_breaks_new_detection_confirmation():
    tracker = SimpleDetectionTracker(new_detection_confirmation_runs=3)

    tracker.preview([_cand(1, 1, "plier", (300, 300, 400, 400))])
    tracker.preview([_cand(2, 1, "plier", (302, 302, 402, 402))])
    tracker.preview([], l2_was_run=True)
    tracker.preview([_cand(4, 1, "plier", (302, 302, 402, 402))])
    result = tracker.preview([_cand(5, 1, "plier", (304, 304, 404, 404))])

    assert not result.l2_new_unmatched_detection


def test_preview_l2_not_run_preserves_new_detection_confirmation():
    tracker = SimpleDetectionTracker(new_detection_confirmation_runs=3)

    tracker.preview([_cand(1, 1, "plier", (300, 300, 400, 400))])
    tracker.preview([], l2_was_run=False)
    tracker.preview([_cand(3, 1, "plier", (302, 302, 402, 402))])
    result = tracker.preview([_cand(4, 1, "plier", (304, 304, 404, 404))])

    assert result.l2_new_unmatched_detection


def test_repeated_preview_for_same_l2_frame_counts_only_once():
    tracker = SimpleDetectionTracker(new_detection_confirmation_runs=3)
    first_run = [_cand(1, 1, "plier", (300, 300, 400, 400))]

    tracker.preview(first_run)
    tracker.preview(first_run)
    second = tracker.preview([
        _cand(2, 1, "plier", (302, 302, 402, 402))
    ])
    third = tracker.preview([
        _cand(3, 1, "plier", (304, 304, 404, 404))
    ])

    assert not second.l2_new_unmatched_detection
    assert third.l2_new_unmatched_detection


def test_preview_accounts_for_inactive_reactivation():
    tracker = SimpleDetectionTracker(
        max_missed_detection_frames=0,
        lost_reactivation_frames=5,
    )
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))], frame_id=0)
    tracker.update([], frame_id=1)

    preview = tracker.preview([
        _cand(2, 0, "wrench", (105, 105, 195, 195)),
    ])

    assert not preview.l2_new_unmatched_detection
    assert preview.unmatched_detection_indices == []


def test_preview_quality_drop_is_repeatable_and_does_not_mutate_state():
    tracker = SimpleDetectionTracker(
        quality_drop_trigger_ratio=0.8,
        quality_drop_min_history=5,
    )
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))])
    tracker._tracks[0].confidence_history = [1.0, 1.0, 1.0, 0.2, 0.2]

    state_before = dict(tracker._track_in_drop)
    detections = [_cand(1, 0, "wrench", (100, 100, 200, 200), confidence=0.2)]
    first = tracker.preview(detections)
    second = tracker.preview(detections)

    assert first.track_quality_drop
    assert second.track_quality_drop
    assert tracker._track_in_drop == state_before


def test_quality_drop_uses_current_detection_and_update_consumes_edge():
    tracker = SimpleDetectionTracker(
        quality_drop_trigger_ratio=0.8,
        quality_drop_rearm_ratio=0.9,
        quality_drop_min_history=5,
    )
    confidences = [1.0, 1.0, 0.1, 0.1]
    for frame_id, confidence in enumerate(confidences):
        tracker.update([
            _cand(
                frame_id,
                0,
                "wrench",
                (100, 100, 200, 200),
                confidence=confidence,
            )
        ], frame_id=frame_id)

    low = _cand(4, 0, "wrench", (100, 100, 200, 200), confidence=0.1)
    assert tracker.preview([low]).track_quality_drop
    assert tracker._track_in_drop == {}

    tracker.update([low], frame_id=4)

    assert tracker._track_in_drop[0]
    assert not tracker.preview([
        _cand(5, 0, "wrench", (100, 100, 200, 200), confidence=0.1)
    ]).track_quality_drop


def test_track_histories_are_bounded():
    tracker = SimpleDetectionTracker(
        max_history_size=3,
        quality_drop_min_history=3,
    )

    for frame_id in range(6):
        tracker.update([
            _cand(
                frame_id,
                0,
                "wrench",
                (100, 100, 200, 200),
                confidence=0.5 + frame_id * 0.01,
            )
        ], frame_id=frame_id)

    track = tracker._tracks[0]
    assert len(track.confidence_history) == 3
    assert len(track.detection_history) == 3
    assert [det.frame_id for det in track.detection_history] == [3, 4, 5]


def test_get_active_tracks():
    tracker = SimpleDetectionTracker()
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))])
    active = tracker.get_active_tracks()
    assert len(active) == 1
    assert active[0].class_name == "wrench"
