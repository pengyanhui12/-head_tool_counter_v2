"""关键帧选择器——触发上下文 → 质量接纳 → RECOVERY

流程:
1. 第一帧无条件接纳
2. 后续帧: 检查触发信号 (max_interval / l2_new_unmatched / track_quality_drop)
3. 无触发 → SKIP
4. 有触发 → 检查 cooldown
5. cooldown 期内 → SKIP（除非 max_interval / end_candidate）
6. 运行 FeatureMatcher.match(current, previous)
7. 匹配成功 (MatchResult.valid=True) → ACCEPTED, 返回 H_current_to_previous
8. 匹配失败 → RECOVERY

注意: Pipeline 不得重复运行 FeatureMatcher——
    KeyframeSelector.evaluate() 已经完成了匹配并返回 H 矩阵。
"""
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
    """关键帧选择器

    职责：判断一帧是否应成为关键帧，如果是则返回已验证的单应矩阵。
    不负责：特征匹配的参数调优（由 FeatureMatcher 负责）。
    """
    def __init__(
        self,
        max_interval: int = 30,
        end_window_frames: int = 30,
        end_best: int = 2,
        end_window_match_candidates: int = 6,
        min_keyframe_interval_frames: int = 5,
        emergency_keyframe_interval_frames: int = 2,
        matcher: FeatureMatcher | None = None,
    ):
        self.max_interval = max_interval
        self.end_window_frames = end_window_frames
        self.end_best = end_best
        self.end_window_match_candidates = int(end_window_match_candidates)
        self.min_keyframe_interval = min_keyframe_interval_frames
        self.emergency_keyframe_interval = emergency_keyframe_interval_frames
        self._matcher = matcher or FeatureMatcher()
        self._last_kf_frame_id: int | None = None
        self._last_kf_gray: np.ndarray | None = None

    def evaluate(
        self,
        frame: Frame,
        previous_keyframe: Frame | BufferedFrame | None,
        trigger_context: KeyframeTriggerContext,
    ) -> KeyframeResult:
        """评估一帧是否应成为关键帧"""
        # 第一帧无条件接纳
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

        # 检查触发信号
        reason = self._check_triggers(frame, trigger_context)
        if reason is None:
            return KeyframeResult(decision=KeyframeDecision.SKIP, reason="")

        # Cooldown 检查
        frames_since_last_kf = frame.frame_id - self._last_kf_frame_id

        # max_interval 和 end_candidate 不受 cooldown 限制
        if reason not in ("max_interval", "end_candidate"):
            if frames_since_last_kf < self.min_keyframe_interval:
                return KeyframeResult(
                    decision=KeyframeDecision.SKIP,
                    reason=f"cooldown_{frames_since_last_kf}",
                )

        # 获取上一关键帧图像用于匹配
        if previous_keyframe is not None:
            prev_img = previous_keyframe.image if hasattr(previous_keyframe, "image") else previous_keyframe.gray
        elif self._last_kf_gray is not None:
            prev_img = self._last_kf_gray
        else:
            prev_img = frame.image

        # 运行特征匹配
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

        # 匹配失败 → RECOVERY
        return KeyframeResult(
            decision=KeyframeDecision.RECOVERY,
            reason=f"match_failed: {match_result.failure_reason}",
            match_result=match_result,
        )

    def select_end_keyframes(
        self,
        end_frames: list[Frame],
    ) -> list[Frame]:
        """从视频尾部窗口中选取最佳 1~2 个关键帧。
        按 frame_id 升序排列以保证正确的 parent 链。
        """
        if not end_frames or self._last_kf_gray is None:
            return sorted(end_frames[: self.end_best], key=lambda f: f.frame_id)

        match_candidates = self._prefilter_end_frames(end_frames)
        scored = []
        for f in match_candidates:
            mr = self._matcher.match(f.image, self._last_kf_gray)
            score = f.sharpness_score * (mr.num_inliers if mr.valid else 0)
            scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return sorted(
            [f for _, f in scored[: self.end_best]],
            key=lambda f: f.frame_id,
        )

    def _prefilter_end_frames(self, end_frames: list[Frame]) -> list[Frame]:
        """按时间分段选择高质量帧，减少尾窗全量SIFT匹配。"""
        ordered = sorted(end_frames, key=lambda frame: frame.frame_id)
        candidate_count = self.end_window_match_candidates
        if candidate_count <= 0 or candidate_count >= len(ordered):
            return ordered

        selected = []
        frame_count = len(ordered)
        for index in range(candidate_count):
            start = index * frame_count // candidate_count
            end = (index + 1) * frame_count // candidate_count
            segment = ordered[start:end]
            # 质量相同时优先更晚帧，确保结果确定并覆盖视频末端。
            selected.append(max(
                segment,
                key=lambda frame: (
                    frame.sharpness_score * frame.exposure_score,
                    frame.sharpness_score,
                    frame.exposure_score,
                    frame.frame_id,
                ),
            ))
        return selected

    def _check_triggers(
        self, frame: Frame, ctx: KeyframeTriggerContext
    ) -> str | None:
        """检查候选触发条件"""
        if ctx.force_end_candidate:
            return "end_candidate"
        if ctx.max_interval_reached:
            return "max_interval"
        if ctx.l2_new_unmatched_detection:
            return "l2_new_unmatched_detection"
        if ctx.track_quality_drop:
            return "track_quality_drop"
        return None
