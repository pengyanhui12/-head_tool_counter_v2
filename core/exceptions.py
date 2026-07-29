"""项目异常类 — 显式语义，便于调试和测试"""
from __future__ import annotations


class TrackBindingConflict(Exception):
    """同一 logical_track_key 尝试绑定到不同 GlobalObject。"""

    def __init__(
        self,
        track_key,
        existing_object_id: str,
        candidate_object_id: str,
        frame_id: int | None = None,
        class_name: str = "",
    ):
        self.track_key = track_key
        self.existing_object_id = existing_object_id
        self.candidate_object_id = candidate_object_id
        self.frame_id = frame_id
        self.class_name = class_name
        msg = (
            f"TrackBindingConflict: logical_track_key={track_key} "
            f"already bound to {existing_object_id}, "
            f"cannot rebind to {candidate_object_id} "
            f"(frame={frame_id}, class={class_name})"
        )
        super().__init__(msg)


class SameFrameObservationError(Exception):
    """同一对象在同一 frame_id 接收了多个 observation。"""

    def __init__(self, object_id: str, frame_id: int):
        self.object_id = object_id
        self.frame_id = frame_id
        super().__init__(
            f"SameFrameObservationError: object {object_id} "
            f"already has an observation in frame {frame_id}"
        )


class MergePolicyError(Exception):
    """合并策略被违反。"""

    def __init__(self, primary_id: str, secondary_id: str, reason: str):
        self.primary_id = primary_id
        self.secondary_id = secondary_id
        self.reason = reason
        super().__init__(
            f"MergePolicyError: cannot merge {secondary_id} into {primary_id}: {reason}"
        )


class InvalidHomographyError(Exception):
    """单应矩阵无效或为 None。"""

    def __init__(self, node_id: int | None = None, reason: str = ""):
        self.node_id = node_id
        self.reason = reason
        super().__init__(f"InvalidHomographyError: node={node_id} {reason}")


class RecoveryError(Exception):
    """恢复过程失败。"""

    def __init__(self, frame_id: int, reason: str):
        self.frame_id = frame_id
        self.reason = reason
        super().__init__(f"RecoveryError: frame={frame_id} {reason}")
