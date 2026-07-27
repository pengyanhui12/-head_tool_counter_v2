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
    tracker = SimpleDetectionTracker()
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))])
    preview = tracker.preview([_cand(1, 1, "plier", (300, 300, 400, 400))])
    assert preview.l2_new_unmatched_detection


def test_get_active_tracks():
    tracker = SimpleDetectionTracker()
    tracker.update([_cand(0, 0, "wrench", (100, 100, 200, 200))])
    active = tracker.get_active_tracks()
    assert len(active) == 1
    assert active[0].class_name == "wrench"
