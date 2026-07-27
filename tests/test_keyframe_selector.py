"""KeyframeSelector 单元测试"""
import numpy as np
import cv2

from core.keyframe_selector import KeyframeSelector
from core.types import Frame, KeyframeDecision, KeyframeTriggerContext


def _make_frame(frame_id: int, seed: int = 0) -> Frame:
    rng = np.random.RandomState(frame_id * 100 + seed)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(30):
        x = rng.randint(20, 620)
        y = rng.randint(20, 460)
        cv2.circle(img, (x, y), rng.randint(3, 8), (255, 255, 255), -1)
    return Frame(frame_id=frame_id, timestamp=frame_id / 30.0, image=img)


def test_first_frame_accepted():
    sel = KeyframeSelector()
    frame = _make_frame(0)
    result = sel.evaluate(frame, previous_keyframe=None,
                          trigger_context=KeyframeTriggerContext())
    assert result.decision == KeyframeDecision.ACCEPTED
    assert result.reason == "first_frame"


def test_no_trigger_skips():
    sel = KeyframeSelector()
    # Accept first
    sel.evaluate(_make_frame(0), previous_keyframe=None,
                 trigger_context=KeyframeTriggerContext())
    # No trigger => skip
    result = sel.evaluate(_make_frame(1), previous_keyframe=_make_frame(0),
                          trigger_context=KeyframeTriggerContext())
    assert result.decision == KeyframeDecision.SKIP


def test_max_interval_triggers():
    sel = KeyframeSelector(max_interval=30)
    sel.evaluate(_make_frame(0), previous_keyframe=None,
                 trigger_context=KeyframeTriggerContext())
    ctx = KeyframeTriggerContext(max_interval_reached=True)
    result = sel.evaluate(_make_frame(31), previous_keyframe=_make_frame(0),
                          trigger_context=ctx)
    assert result.decision != KeyframeDecision.SKIP


def test_force_end_candidate():
    sel = KeyframeSelector()
    sel.evaluate(_make_frame(0), previous_keyframe=None,
                 trigger_context=KeyframeTriggerContext())
    ctx = KeyframeTriggerContext(force_end_candidate=True)
    result = sel.evaluate(_make_frame(35), previous_keyframe=_make_frame(0),
                          trigger_context=ctx)
    assert result.decision != KeyframeDecision.SKIP
