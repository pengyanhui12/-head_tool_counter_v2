"""关键帧选择器——触发上下文 → 质量接纳 → RECOVERY

流程:
1. 第一帧无条件接纳
2. 后续帧: 检查触发信号 (max_interval / l2_new_unmatched / track_quality_drop)
3. 无触发 → SKIP
4. 有触发 → 运行 FeatureMatcher.match(current, previous)
5. 匹配成功 (MatchResult.valid=True) → ACCEPTED, 返回 H_current_to_previous
6. 匹配失败 → RECOVERY

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
        max_interval: int = 30,            # 最大关键帧间隔（帧数）
        end_window_frames: int = 30,       # 视频尾窗口大小
        end_best: int = 2,                 # 尾部选取最佳帧数
        matcher: FeatureMatcher | None = None,  # 特征匹配器，默认创建 SIFT 实例
    ):
        self.max_interval = max_interval
        self.end_window_frames = end_window_frames
        self.end_best = end_best
        self._matcher = matcher or FeatureMatcher()
        self._last_kf_frame_id: int | None = None  # 上一个关键帧的 frame_id
        self._last_kf_gray: np.ndarray | None = None  # 上一个关键帧的灰度图

    def evaluate(
        self,
        frame: Frame,
        previous_keyframe: Frame | BufferedFrame | None,
        trigger_context: KeyframeTriggerContext,
    ) -> KeyframeResult:
        """评估一帧是否应成为关键帧

        Args:
            frame: 当前帧 (Frame)
            previous_keyframe: 上一个已接纳的关键帧 (Frame 或 BufferedFrame)
            trigger_context: 来自 L2 + Tracker 的触发信号汇总

        Returns:
            KeyframeResult: 包含决策 (SKIP/ACCEPTED/RECOVERY) 和 H_current_to_previous 矩阵
        """
        # 第一帧无条件接纳
        if self._last_kf_frame_id is None:
            self._last_kf_frame_id = frame.frame_id
            if previous_keyframe is not None:
                # 提取灰度图缓存，加速后续匹配
                pimg = previous_keyframe.image if hasattr(previous_keyframe, "image") else previous_keyframe.gray
                self._last_kf_gray = pimg
            return KeyframeResult(
                decision=KeyframeDecision.ACCEPTED,
                reason="first_frame",
                H_current_to_previous=np.eye(3),  # 第一帧 H=I
            )

        # 检查触发信号——无触发则跳过
        reason = self._check_triggers(frame, trigger_context)
        if reason is None:
            return KeyframeResult(decision=KeyframeDecision.SKIP, reason="")

        # 获取上一关键帧图像用于匹配
        if previous_keyframe is not None:
            prev_img = previous_keyframe.image if hasattr(previous_keyframe, "image") else previous_keyframe.gray
        elif self._last_kf_gray is not None:
            prev_img = self._last_kf_gray
        else:
            prev_img = frame.image

        # ---- 核心: 运行特征匹配 ----
        # source=当前帧(current), target=上一关键帧(previous)
        # → 返回 H_current_to_previous (将当前帧坐标映射到上一帧坐标)
        match_result = self._matcher.match(frame.image, prev_img)

        if match_result.valid and match_result.H_source_to_target is not None:
            # 匹配成功 → 接纳为关键帧
            self._last_kf_frame_id = frame.frame_id
            self._last_kf_gray = frame.image
            return KeyframeResult(
                decision=KeyframeDecision.ACCEPTED,
                reason=reason,
                H_current_to_previous=match_result.H_source_to_target,
                match_result=match_result,
            )

        # 匹配失败 → RECOVERY（MVP 暂不实现过渡帧查找和历史重定位）
        return KeyframeResult(
            decision=KeyframeDecision.RECOVERY,
            reason=f"match_failed: {match_result.failure_reason}",
            match_result=match_result,
        )

    def select_end_keyframes(
        self,
        end_frames: list[Frame],
    ) -> list[Frame]:
        """从视频尾部窗口中选取最佳 1~2 个关键帧

        评分: sharpness_score * num_inliers (仅 valid 匹配计入)
        """
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
        """检查候选触发条件，返回触发原因字符串或 None

        当前支持的触发:
        - force_end_candidate: 视频尾帧强触发
        - max_interval_reached: 距上一关键帧超过 max_interval 帧
        - l2_new_unmatched_detection: L2 检测到与已有 track 不匹配的新目标
        - track_quality_drop: 活跃 track 置信度突降 >30%
        """
        if ctx.force_end_candidate:
            return "end_candidate"
        if ctx.max_interval_reached:
            return "max_interval"
        if ctx.l2_new_unmatched_detection:
            return "l2_new_unmatched_detection"
        if ctx.track_quality_drop:
            return "track_quality_drop"
        return None
