"""关键帧选择器——触发上下文 → 质量接纳 → RECOVERY"""
import numpy as np

from core.types import (
    Frame,
    BufferedFrame,
    KeyframeDecision,
    KeyframeResult,
    KeyframeTriggerContext,
)
from core.feature_matcher import FeatureMatcher


class KeyframeSelector:
    def __init__(
        self,
        max_interval: int = 30,
        end_window_frames: int = 30,
        end_best: int = 2,
        matcher: FeatureMatcher | None = None,
    ):
        self.max_interval = max_interval
        self.end_window_frames = end_window_frames
        self.end_best = end_best
        self._matcher = matcher or FeatureMatcher()
        self._last_kf_frame_id: int | None = None
        self._last_kf_gray: np.ndarray | None = None

    def evaluate(
        self,
        frame: Frame,
        previous_keyframe: Frame | BufferedFrame | None,
        trigger_context: KeyframeTriggerContext,
    ) -> KeyframeResult:
        # First frame always accepted
        if self._last_kf_frame_id is None:
            self._last_kf_frame_id = frame.frame_id
            if previous_keyframe is not None:
                pimg = previous_keyframe.image if hasattr(previous_keyframe, "image") else previous_keyframe.gray
                self._last_kf_gray = pimg
            return KeyframeResult(
                decision=KeyframeDecision.ACCEPTED,
                reason="first_frame",
                H_current_to_previous=np.eye(3),
            )

        # Check triggers
        reason = self._check_triggers(frame, trigger_context)
        if reason is None:
            return KeyframeResult(decision=KeyframeDecision.SKIP, reason="")

        # Get previous keyframe image for matching
        if previous_keyframe is not None:
            prev_img = previous_keyframe.image if hasattr(previous_keyframe, "image") else previous_keyframe.gray
        elif self._last_kf_gray is not None:
            prev_img = self._last_kf_gray
        else:
            prev_img = frame.image

        # Run feature matching
        # source=current, target=previous => H_current_to_previous
        match_result = self._matcher.match(frame.image, prev_img)

        if match_result.valid and match_result.H_source_to_target is not None:
            self._last_kf_frame_id = frame.frame_id
            self._last_kf_gray = frame.image
            return KeyframeResult(
                decision=KeyframeDecision.ACCEPTED,
                reason=reason,
                H_current_to_previous=match_result.H_source_to_target,
                match_result=match_result,
            )

        # RECOVERY: try transition frames from FrameBuffer? Not in MVP.
        # For now, return RECOVERY.
        return KeyframeResult(
            decision=KeyframeDecision.RECOVERY,
            reason=f"match_failed: {match_result.failure_reason}",
            match_result=match_result,
        )

    def select_end_keyframes(
        self,
        end_frames: list[Frame],
    ) -> list[Frame]:
        if not end_frames or self._last_kf_gray is None:
            return end_frames[: self.end_best]

        scored = []
        for f in end_frames:
            mr = self._matcher.match(f.image, self._last_kf_gray)
            score = f.sharpness_score * (mr.num_inliers if mr.valid else 0)
            scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[: self.end_best]]

    def _check_triggers(
        self, frame: Frame, ctx: KeyframeTriggerContext
    ) -> str | None:
        if ctx.force_end_candidate:
            return "end_candidate"
        if ctx.max_interval_reached:
            return "max_interval"
        if ctx.l2_new_unmatched_detection:
            return "l2_new_unmatched_detection"
        if ctx.track_quality_drop:
            return "track_quality_drop"
        return None
