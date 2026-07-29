"""恢复管理器 — RECOVERY 路径：bridge → history → LOST

禁止使用单位阵伪造定位。
"""
from __future__ import annotations

import numpy as np

from core.types import (
    Frame, BufferedFrame, MatchResult, RecoveryResult, RecoveryState,
)


class RecoveryManager:
    """关键帧匹配失败后的恢复逻辑。

    策略：
    1. bridge: 尝试 current → buffer 中最近质量好的帧 → previous
    2. history_anchor: 尝试 current → 图中更早的节点
    3. LOST: 全部失败，不加图节点、不投影、不关联

    禁止使用 np.eye(3) 作为 fallback。
    """

    def __init__(self, matcher=None, buffer_size: int = 6):
        self._matcher = matcher
        self._buffer_size = buffer_size
        self._recovery_buffer: list[Frame] = []
        self._cached_detections: dict[int, list] = {}
        self._attempts: int = 0
        self._successes: int = 0
        self._failures: int = 0

    def reset(self):
        """成功关键帧后清空 buffer 和缓存。"""
        self._recovery_buffer.clear()
        self._cached_detections.clear()

    def cache_frame(self, frame: Frame):
        """将跳过的帧加入 buffer（最多 buffer_size 帧）。"""
        self._recovery_buffer.append(frame)
        if len(self._recovery_buffer) > self._buffer_size:
            self._recovery_buffer.pop(0)

    def cache_detections(self, frame_id: int, detections: list, sharpness: float):
        """缓存检测结果（质量差的帧恢复后可能用到）。"""
        self._cached_detections[frame_id] = {
            "detections": detections,
            "sharpness": sharpness,
        }

    def recover(
        self,
        current_frame: Frame,
        previous_keyframe: Frame | BufferedFrame | None,
        frame_buffer=None,
        graph=None,
        keyframe_images: dict | None = None,
    ) -> RecoveryResult:
        """尝试恢复。

        Returns:
            RecoveryResult with state, anchor_node_id, H_current_to_anchor
        """
        self._attempts += 1

        if self._matcher is None:
            self._failures += 1
            return RecoveryResult(state=RecoveryState.LOST)

        # Strategy 1: bridge recovery
        if self._recovery_buffer and previous_keyframe is not None:
            best_frame = self._select_best_buffer_frame()
            if best_frame is not None:
                mr1 = self._matcher.match(current_frame.image, best_frame.image)
                if mr1.valid and mr1.H_source_to_target is not None:
                    # current → bridge
                    H_curr_to_bridge = mr1.H_source_to_target
                    # bridge → previous
                    prev_img = previous_keyframe.image if hasattr(previous_keyframe, "image") else previous_keyframe.gray
                    mr2 = self._matcher.match(best_frame.image, prev_img)
                    if mr2.valid and mr2.H_source_to_target is not None:
                        H_bridge_to_prev = mr2.H_source_to_target
                        # chain: current → bridge → previous
                        H_curr_to_prev = H_bridge_to_prev @ H_curr_to_bridge
                        self._successes += 1
                        return RecoveryResult(
                            state=RecoveryState.RECOVERED,
                            anchor_node_id=None,
                            H_current_to_anchor=H_curr_to_prev,
                            match_result=mr1,
                        )

        # Strategy 2: history anchor
        if graph is not None and keyframe_images is not None:
            node_ids = sorted(keyframe_images.keys(), reverse=True)
            # 跳过最近2个节点（它们可能就是匹配失败的原因）
            for anchor_id in node_ids[2:5]:
                anchor_img = keyframe_images.get(anchor_id)
                if anchor_img is None:
                    continue
                mr = self._matcher.match(current_frame.image, anchor_img)
                if mr.valid and mr.H_source_to_target is not None:
                    self._successes += 1
                    return RecoveryResult(
                        state=RecoveryState.RECOVERED,
                        anchor_node_id=anchor_id,
                        H_current_to_anchor=mr.H_source_to_target,
                        match_result=mr,
                    )

        self._failures += 1
        return RecoveryResult(state=RecoveryState.LOST)

    def _select_best_buffer_frame(self) -> Frame | None:
        if not self._recovery_buffer:
            return None
        return max(self._recovery_buffer, key=lambda f: f.sharpness_score, default=None)

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def successes(self) -> int:
        return self._successes

    @property
    def failures(self) -> int:
        return self._failures
