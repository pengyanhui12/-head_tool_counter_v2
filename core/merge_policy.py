"""合并策略 — 集中所有合并安全条件检查

禁止的合并模式:
1. 同帧共现 → 永久禁止合并
2. observation frame 有重叠 → 禁止合并
3. track conflict → 禁止合并
4. 不同类别且不兼容 → 禁止合并

所有合并必须产生 MergeAudit 记录。
"""
from __future__ import annotations

from core.types import GlobalObject, GlobalDetection, MergeAudit


class MergePolicy:
    """合并策略检查器。不执行合并，只判定是否允许。"""

    def __init__(self):
        self._co_occurred_pairs: set[frozenset[str]] = set()
        self._frame_co_occurred: set[tuple[int, str, str]] = set()

    def record_co_occurrence(self, frame_id: int, pid_a: str, pid_b: str):
        """记录两个对象在同帧共现。"""
        if pid_a == pid_b:
            return
        pair = frozenset([pid_a, pid_b])
        self._co_occurred_pairs.add(pair)
        self._frame_co_occurred.add((frame_id, pid_a, pid_b))

    def have_co_occurred(self, pid_a: str, pid_b: str) -> bool:
        """两个对象是否曾在任何帧共现过。"""
        return frozenset([pid_a, pid_b]) in self._co_occurred_pairs

    def can_merge(
        self,
        primary: GlobalObject,
        secondary: GlobalObject,
        shared_track_keys: list = None,
    ) -> tuple[bool, str, MergeAudit | None]:
        """检查是否可以合并 secondary → primary。

        Returns:
            (allowed, reason_or_block_reason, audit_or_None)
        """
        shared_track_keys = shared_track_keys or []

        # 1. 类别兼容性
        if primary.class_name != secondary.class_name:
            audit = MergeAudit(
                primary_id=primary.provisional_id,
                secondary_id=secondary.provisional_id,
                decision="blocked",
                reason="class_mismatch",
                shared_track_keys=shared_track_keys,
            )
            return False, "class_mismatch", audit

        # 2. 同帧共现检查
        pair = frozenset([primary.provisional_id, secondary.provisional_id])
        if pair in self._co_occurred_pairs:
            audit = MergeAudit(
                primary_id=primary.provisional_id,
                secondary_id=secondary.provisional_id,
                decision="blocked",
                reason="co_occurrence",
                co_occurred=True,
                shared_track_keys=shared_track_keys,
            )
            return False, "co_occurrence", audit

        # 3. observation frame 重叠检查
        primary_frames = {obs.frame_id for obs in primary.observations}
        secondary_frames = {obs.frame_id for obs in secondary.observations}
        overlap = primary_frames & secondary_frames
        if overlap:
            audit = MergeAudit(
                primary_id=primary.provisional_id,
                secondary_id=secondary.provisional_id,
                decision="blocked",
                reason="observation_frame_overlap",
                overlapping_frame_ids=sorted(overlap),
                shared_track_keys=shared_track_keys,
            )
            return False, "observation_frame_overlap", audit

        # 4. track conflict 检查
        if ReviewFlag.TRACK_CONFLICT in primary.review_flags:
            audit = MergeAudit(
                primary_id=primary.provisional_id,
                secondary_id=secondary.provisional_id,
                decision="blocked",
                reason="primary_has_track_conflict",
                shared_track_keys=shared_track_keys,
            )
            return False, "primary_has_track_conflict", audit
        if ReviewFlag.TRACK_CONFLICT in secondary.review_flags:
            audit = MergeAudit(
                primary_id=primary.provisional_id,
                secondary_id=secondary.provisional_id,
                decision="blocked",
                reason="secondary_has_track_conflict",
                shared_track_keys=shared_track_keys,
            )
            return False, "secondary_has_track_conflict", audit

        # 5. 允许合并
        import numpy as np
        pos_dist = float(np.linalg.norm(
            np.array(primary.centroid_xy) - np.array(secondary.centroid_xy)
        ))
        audit = MergeAudit(
            primary_id=primary.provisional_id,
            secondary_id=secondary.provisional_id,
            decision="merged",
            reason="shared_track" if shared_track_keys else "manual",
            shared_track_keys=shared_track_keys,
            position_distance=pos_dist,
            co_occurred=False,
        )
        return True, "allowed", audit


# Need to import ReviewFlag at module level
from core.types import ReviewFlag
