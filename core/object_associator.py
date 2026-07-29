"""对象关联器 — 门控 + 匈牙利匹配 + 多帧投票 + 审计合并

不变量:
- O1: 一个 logical_track_key 在未发生合法合并前只能绑定一个对象
- O2: 一个对象在同一 frame_id 最多一个 observation
- O3: 同一帧不同融合检测必须对应不同对象
- O4: 曾同帧共现的对象永久禁止自动合并
- O5: 合并前两个对象的 observation frame 集合不得重叠
- O6: 每个 REJECTED 必须有 rejected_reason
- O7: 合并产生的 REJECTED 必须有 merged_into_id
- O8: observation_count == len(observations)
- O9: 每个对象 observation frame_id 唯一
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from core.types import (
    GlobalDetection,
    GlobalObject,
    ConfirmationStatus,
    VisibilityStatus,
    ReviewFlag,
    RebuildResult,
    MergeAudit,
)
from core.global_object_map import GlobalObjectMap
from core.exceptions import TrackBindingConflict, SameFrameObservationError
from core.merge_policy import MergePolicy


class ObjectAssociator:
    def __init__(
        self,
        max_position_distance_px: float = 120.0,
        position_weight: float = 0.55,
        overlap_weight: float = 0.20,
        size_weight: float = 0.10,
        class_weight: float = 0.15,
        max_cost: float = 0.75,
        min_observations_confirmed: int = 3,
        min_keyframes_confirmed: int = 2,
        min_top_class_ratio: float = 0.60,
        max_votes_per_track: int = 3,
        class_compatibility: dict[str, list[str]] | None = None,
        debug_mode: bool = False,
    ):
        self.max_position_distance = max_position_distance_px
        self.w_pos = position_weight
        self.w_overlap = overlap_weight
        self.w_size = size_weight
        self.w_class = class_weight
        self.max_cost = max_cost
        self.min_obs_confirm = min_observations_confirmed
        self.min_kf_confirm = min_keyframes_confirmed
        self.min_top_class_ratio = min_top_class_ratio
        self.max_votes_per_track = max_votes_per_track
        self._class_compat = class_compatibility or {}
        self._debug_mode = debug_mode

        self.map = GlobalObjectMap()
        self._all_gd: list[GlobalDetection] = []
        self._track_vote_counts: dict[tuple[int, int], int] = {}
        self._track_to_object: dict[tuple[int, int], str] = {}

        # 共现记录
        self._co_occurred_pairs: set[frozenset[str]] = set()
        self._frame_co_occurred: set[tuple[int, str, str]] = set()

        # 合并策略
        self._merge_policy = MergePolicy()

        # 合并审计
        self._merge_audits: list[MergeAudit] = []

        # 统计
        self.stats = {
            "objects_created": 0,
            "objects_matched_by_track": 0,
            "objects_matched_by_cost": 0,
            "objects_merged": 0,
            "merge_blocked_by_cooccurrence": 0,
            "merge_blocked_by_frame_overlap": 0,
            "track_binding_conflicts": 0,
            "same_frame_duplicate_observations": 0,
        }

    # ── 在线帧处理 ──

    def ingest_frame(
        self, frame_id: int, global_detections: list[GlobalDetection]
    ) -> list[str]:
        """处理一帧的全局检测，关联到已有对象或创建新对象。

        匹配优先级：
        1. track logical_key 强关联
        2. 帧级全局匈牙利（空间 + 类别 + 尺寸）
        3. 创建新对象

        不变量：同一帧内每个对象最多一个 observation，
        每个 detection 最多分配给一个对象。
        """
        affected: list[str] = []
        if not global_detections:
            return affected

        assigned_object_ids: set[str] = set()
        assigned_detection_indices: set[int] = set()
        unmatched_gds: list[tuple[int, GlobalDetection]] = []

        for di, gd in enumerate(global_detections):
            self._all_gd.append(gd)
            logical_key = self._make_logical_key(gd.track_id)

            existing_pid = self._track_to_object.get(logical_key)
            if existing_pid is not None:
                # 同帧内一个对象只能匹配一次
                if existing_pid in assigned_object_ids:
                    continue
                obj = self.map.get_by_provisional(existing_pid)
                if obj is not None and self._class_compatible(obj.class_name, gd.class_name):
                    # 检查同帧重复 observation
                    if self._has_observation_in_frame(obj, frame_id):
                        self.stats["same_frame_duplicate_observations"] += 1
                        if self._debug_mode:
                            raise SameFrameObservationError(obj.provisional_id, frame_id)
                        continue
                    self._update_object(obj, gd)
                    affected.append(obj.provisional_id)
                    assigned_object_ids.add(obj.provisional_id)
                    assigned_detection_indices.add(di)
                    self.stats["objects_matched_by_track"] += 1
                    continue
            unmatched_gds.append((di, gd))

        if not unmatched_gds:
            self._prune(frame_id)
            return affected

        all_objects = self.map.get_all()
        if not all_objects:
            for di, gd in unmatched_gds:
                if di in assigned_detection_indices:
                    continue
                obj = self.map.create_object(gd)
                logical_key = self._make_logical_key(gd.track_id)
                self._bind_track_to_object(logical_key, obj.provisional_id)
                affected.append(obj.provisional_id)
                assigned_object_ids.add(obj.provisional_id)
                assigned_detection_indices.add(di)
                self.stats["objects_created"] += 1
            self._prune(frame_id)
            return affected

        # 帧级全局匈牙利
        gd_list = [gd for _, gd in unmatched_gds]
        n_det = len(gd_list)
        n_obj = len(all_objects)
        global_cost = np.full((n_det, n_obj), 1e9)

        for di, gd in enumerate(gd_list):
            for oi, obj in enumerate(all_objects):
                if obj.provisional_id in assigned_object_ids:
                    continue
                if self._has_observation_in_frame(obj, frame_id):
                    continue
                if not self._class_compatible(obj.class_name, gd.class_name):
                    continue
                pos_dist = np.linalg.norm(
                    np.array(gd.polygon_centroid) - np.array(obj.centroid_xy)
                )
                online_gate = self.max_position_distance * 0.4
                if pos_dist > online_gate:
                    continue
                area_min, area_max = obj.area_range
                size_cost = (abs(gd.polygon_area - (area_min + area_max) / 2) / area_max
                             if area_max > 0 else 0.0)
                class_cost = self._class_cost(obj.class_name, gd.class_name)
                global_cost[di, oi] = (
                    self.w_pos * pos_dist / self.max_position_distance
                    + self.w_size * min(size_cost, 1.0)
                    + self.w_class * class_cost
                )

        ri, ci = linear_sum_assignment(global_cost)
        for r, c in zip(ri, ci):
            if global_cost[r, c] >= 1e9 or global_cost[r, c] > self.max_cost:
                continue
            obj = all_objects[c]
            if obj.provisional_id in assigned_object_ids:
                continue
            gd = gd_list[r]
            if self._has_observation_in_frame(obj, frame_id):
                self.stats["same_frame_duplicate_observations"] += 1
                continue
            self._update_object(obj, gd)
            affected.append(obj.provisional_id)
            assigned_object_ids.add(obj.provisional_id)
            self.stats["objects_matched_by_cost"] += 1

        # 创建新对象
        for di, gd in unmatched_gds:
            gd_idx = unmatched_gds.index((di, gd))
            if gd_idx >= len(gd_list):
                continue
            already_matched = False
            for r in ri:
                if global_cost[r, ci[list(ri).index(r)]] < 1e9 and list(ri).index(r) == gd_idx:
                    already_matched = True
                    break
            if already_matched:
                continue
            obj = self.map.create_object(gd)
            logical_key = self._make_logical_key(gd.track_id)
            self._bind_track_to_object(logical_key, obj.provisional_id)
            affected.append(obj.provisional_id)
            assigned_object_ids.add(obj.provisional_id)
            self.stats["objects_created"] += 1

        # 记录共现
        self._record_co_occurrences(frame_id, global_detections)

        # 验证不变量
        for obj in self.map.get_all():
            frame_ids = [obs.frame_id for obs in obj.observations]
            if len(frame_ids) != len(set(frame_ids)):
                self.stats["same_frame_duplicate_observations"] += 1

        self._prune(frame_id)
        return affected

    def _record_co_occurrences(
        self, frame_id: int, global_detections: list[GlobalDetection]
    ) -> None:
        """记录本帧内不同对象间的共现关系。"""
        pid_per_det: list[tuple[GlobalDetection, str | None]] = []
        for gd in global_detections:
            logical_key = self._make_logical_key(gd.track_id)
            pid = self._track_to_object.get(logical_key)
            if pid is None:
                pid = self._find_pid_for_detection(gd)
            pid_per_det.append((gd, pid))

        for i, (gd_a, pid_a) in enumerate(pid_per_det):
            if pid_a is None:
                continue
            for j, (gd_b, pid_b) in enumerate(pid_per_det):
                if i >= j or pid_b is None or pid_a == pid_b:
                    continue
                pair = frozenset([pid_a, pid_b])
                if pair in self._co_occurred_pairs:
                    continue
                if gd_a.class_name != gd_b.class_name:
                    continue
                if gd_a.bbox_pixels and gd_b.bbox_pixels:
                    iou = self._compute_bbox_iou(gd_a.bbox_pixels, gd_b.bbox_pixels)
                    if iou == 0.0:
                        self._frame_co_occurred.add((frame_id, pid_a, pid_b))
                        self._co_occurred_pairs.add(pair)
                        self._merge_policy.record_co_occurrence(frame_id, pid_a, pid_b)

    # ── 后处理合并（final_review）──

    def final_review(self) -> None:
        """后处理：track_id 硬合并 + 审计标记。

        规则：
        1. shared track + 全部安全条件满足 → 合并
        2. centroid < 30px → 只标记 LIKELY_DUPLICATE，不自动合并
        3. shared track 但有 co-occurrence → 不合并，标记 LIKELY_DUPLICATE
        4. 所有 REJECTED 必须有 rejected_reason
        """
        self._merge_by_shared_track_safe()
        self._mark_close_duplicates()

    def _merge_by_shared_track_safe(self) -> None:
        """仅当 shared track + 全部安全条件满足时才合并。"""
        objs = [o for o in self.map.get_all()
                if o.confirmation_status in (ConfirmationStatus.CONFIRMED, ConfirmationStatus.TENTATIVE)]
        by_class: dict[str, list[GlobalObject]] = {}
        for o in objs:
            by_class.setdefault(o.class_name, []).append(o)

        for _class_name, group in by_class.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda o: o.observation_count, reverse=True)
            merged: set[str] = set()
            for i, primary in enumerate(group):
                if primary.provisional_id in merged:
                    continue
                for secondary in group[i + 1:]:
                    if secondary.provisional_id in merged:
                        continue
                    shared_tracks = list(primary.track_ids & secondary.track_ids)
                    if not shared_tracks:
                        continue

                    allowed, reason, audit = self._merge_policy.can_merge(
                        primary, secondary, shared_track_keys=shared_tracks,
                    )
                    if audit:
                        self._merge_audits.append(audit)

                    if allowed:
                        self._merge_objects(primary, secondary)
                        merged.add(secondary.provisional_id)
                        self._reevaluate(primary)
                        self.stats["objects_merged"] += 1
                    else:
                        if reason == "co_occurrence":
                            self.stats["merge_blocked_by_cooccurrence"] += 1
                        elif reason == "observation_frame_overlap":
                            self.stats["merge_blocked_by_frame_overlap"] += 1
                        primary.review_flags.add(ReviewFlag.LIKELY_DUPLICATE)
                        secondary.review_flags.add(ReviewFlag.LIKELY_DUPLICATE)

    def _mark_close_duplicates(self) -> None:
        """质心接近但无 shared track → 仅标记 LIKELY_DUPLICATE，不自动合并。"""
        objs = [o for o in self.map.get_all()
                if o.confirmation_status in (ConfirmationStatus.CONFIRMED, ConfirmationStatus.TENTATIVE)]
        by_class: dict[str, list[GlobalObject]] = {}
        for o in objs:
            by_class.setdefault(o.class_name, []).append(o)

        for _class_name, group in by_class.items():
            if len(group) < 2:
                continue
            for i, primary in enumerate(group):
                for secondary in group[i + 1:]:
                    pair = frozenset([primary.provisional_id, secondary.provisional_id])
                    if pair in self._co_occurred_pairs:
                        continue
                    dist = np.linalg.norm(
                        np.array(primary.centroid_xy) - np.array(secondary.centroid_xy)
                    )
                    if dist < 30.0:
                        primary.review_flags.add(ReviewFlag.LIKELY_DUPLICATE)
                        secondary.review_flags.add(ReviewFlag.LIKELY_DUPLICATE)

    def _merge_objects(self, primary: GlobalObject, secondary: GlobalObject) -> None:
        """将 secondary 合并到 primary，secondary 标 REJECTED 并记录审计信息。"""
        # O5: 合并前检查 frame overlap
        p_frames = {obs.frame_id for obs in primary.observations}
        s_frames = {obs.frame_id for obs in secondary.observations}
        if p_frames & s_frames:
            self.stats["merge_blocked_by_frame_overlap"] += 1
            return

        w1, w2 = primary.observation_count, secondary.observation_count
        primary.observations.extend(secondary.observations)
        primary.observation_count = len(primary.observations)
        primary.track_ids.update(secondary.track_ids)
        primary.keyframe_ids.update(secondary.keyframe_ids)
        for cls_name, votes in secondary.vote_distribution.items():
            primary.vote_distribution[cls_name] = (
                primary.vote_distribution.get(cls_name, 0.0) + votes
            )
        tw = w1 + w2
        if tw > 0:
            primary.centroid_xy = (
                float((primary.centroid_xy[0] * w1 + secondary.centroid_xy[0] * w2) / tw),
                float((primary.centroid_xy[1] * w1 + secondary.centroid_xy[1] * w2) / tw),
            )

        # 转移 track 绑定
        for track_id in list(secondary.track_ids):
            logical_key = self._make_logical_key(track_id)
            old_binding = self._track_to_object.get(logical_key)
            if old_binding == secondary.provisional_id:
                self._track_to_object[logical_key] = primary.provisional_id

        # O6 + O7: REJECTED 必须记录原因
        secondary.confirmation_status = ConfirmationStatus.REJECTED
        secondary.rejected_reason = "merged_duplicate"
        secondary.merged_into_id = primary.provisional_id
        secondary.rejection_evidence = {
            "shared_tracks": list(primary.track_ids & secondary.track_ids),
            "merged_at_obs_count": primary.observation_count,
        }
        secondary.visibility_status = VisibilityStatus.INACTIVE

        self.stats["objects_merged"] += 1

    def validate_object_map(self) -> dict:
        """验证所有不变量，返回违规报告。"""
        violations = {
            "missing_rejected_reason": [],
            "missing_merged_into": [],
            "duplicate_frame_observations": [],
            "track_binding_conflicts": [],
            "persistent_id_on_rejected": [],
        }

        active_pids = set()
        for obj in self.map.get_all():
            if obj.confirmation_status == ConfirmationStatus.REJECTED:
                if obj.rejected_reason is None:
                    violations["missing_rejected_reason"].append(obj.provisional_id)
                if obj.merged_into_id is None and "merged" in (obj.rejected_reason or ""):
                    violations["missing_merged_into"].append(obj.provisional_id)
                if obj.persistent_id is not None:
                    violations["persistent_id_on_rejected"].append(obj.provisional_id)
            else:
                active_pids.add(obj.provisional_id)

            # O9: observation frame 唯一
            frame_ids = [obs.frame_id for obs in obj.observations]
            if len(frame_ids) != len(set(frame_ids)):
                violations["duplicate_frame_observations"].append({
                    "provisional_id": obj.provisional_id,
                    "observation_count": obj.observation_count,
                    "unique_frames": len(set(frame_ids)),
                })

            # O8: observation_count == len(observations)
            if obj.observation_count != len(obj.observations):
                violations["duplicate_frame_observations"].append({
                    "provisional_id": obj.provisional_id,
                    "reported_count": obj.observation_count,
                    "actual_count": len(obj.observations),
                })

        return violations

    def get_reportable_objects(self) -> list[GlobalObject]:
        """返回所有非 REJECTED 的对象。"""
        return [
            o for o in self.map.get_all()
            if o.confirmation_status != ConfirmationStatus.REJECTED
        ]

    @property
    def merge_audits(self) -> list[MergeAudit]:
        return list(self._merge_audits)

    # ── private helpers ──

    def _make_logical_key(self, track_id: int, generation: int = 0) -> tuple[int, int]:
        return (track_id, generation)

    def _has_observation_in_frame(self, obj: GlobalObject, frame_id: int) -> bool:
        return any(obs.frame_id == frame_id for obs in obj.observations)

    def _bind_track_to_object(
        self, logical_track_key: tuple[int, int], object_id: str
    ) -> None:
        """绑定 logical_track_key → object_id。
        如果已有不同绑定，抛出 TrackBindingConflict（不覆盖）。
        """
        existing = self._track_to_object.get(logical_track_key)
        if existing is None:
            self._track_to_object[logical_track_key] = object_id
            return
        if existing == object_id:
            return
        # 冲突：不覆盖，抛出异常或记录
        self.stats["track_binding_conflicts"] += 1
        if self._debug_mode:
            raise TrackBindingConflict(
                track_key=logical_track_key,
                existing_object_id=existing,
                candidate_object_id=object_id,
            )
        # 非 debug 模式：标记冲突但不覆盖
        existing_obj = self.map.get_by_provisional(existing)
        candidate_obj = self.map.get_by_provisional(object_id)
        if existing_obj:
            existing_obj.review_flags.add(ReviewFlag.TRACK_CONFLICT)
        if candidate_obj:
            candidate_obj.review_flags.add(ReviewFlag.TRACK_CONFLICT)

    def _find_pid_for_detection(self, gd: GlobalDetection) -> str | None:
        """用 frame_id + track_id 查找检测归属的 object provisional_id。"""
        for obj in self.map.get_all():
            for obs in obj.observations:
                if obs.frame_id == gd.frame_id and obs.track_id == gd.track_id:
                    return obj.provisional_id
            if gd.track_id in obj.track_ids:
                return obj.provisional_id
        return None

    @staticmethod
    def _compute_bbox_iou(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        xa, ya, xa2, ya2 = a
        xb, yb, xb2, yb2 = b
        xi1, yi1 = max(xa, xb), max(ya, yb)
        xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        area_a = max(1.0, (xa2 - xa) * (ya2 - ya))
        area_b = max(1.0, (xb2 - xb) * (yb2 - yb))
        return inter / (area_a + area_b - inter)

    def _update_object(self, obj: GlobalObject, gd: GlobalDetection) -> None:
        """更新对象，累计投票权重。"""
        obj.observations.append(gd)
        obj.observation_count = len(obj.observations)  # O8
        obj.track_ids.add(gd.track_id)
        obj.keyframe_ids.add(gd.keyframe_id)
        alpha = 0.3
        obj.centroid_xy = (
            float(alpha * gd.polygon_centroid[0] + (1 - alpha) * obj.centroid_xy[0]),
            float(alpha * gd.polygon_centroid[1] + (1 - alpha) * obj.centroid_xy[1]),
        )
        areas = [obs.polygon_area for obs in obj.observations if obs.polygon_area > 0]
        if areas:
            obj.area_range = (min(areas), max(areas))

        logical_key = self._make_logical_key(gd.track_id)
        votes_used = self._track_vote_counts.get(logical_key, 0)
        if votes_used < self.max_votes_per_track:
            weight = (
                gd.detection_confidence
                * gd.sharpness / 100.0
                * gd.mapping_quality
                * gd.edge_quality
                * gd.size_quality
            )
            obj.vote_distribution[gd.class_name] = (
                obj.vote_distribution.get(gd.class_name, 0.0) + weight
            )
            self._track_vote_counts[logical_key] = votes_used + 1

        self._reevaluate(obj)

    def _reevaluate(self, obj: GlobalObject) -> None:
        if (obj.observation_count >= self.min_obs_confirm
                and len(obj.keyframe_ids) >= self.min_kf_confirm):
            total_votes = sum(obj.vote_distribution.values())
            if total_votes > 0:
                top_class = max(obj.vote_distribution, key=obj.vote_distribution.get)
                top_ratio = obj.vote_distribution[top_class] / total_votes
                if top_ratio >= self.min_top_class_ratio:
                    obj.confirmation_status = ConfirmationStatus.CONFIRMED
                    obj.confidence = top_ratio
                else:
                    obj.confirmation_status = ConfirmationStatus.UNCERTAIN
                    obj.uncertainty_reasons.append("low_class_consensus")

    def _prune(self, current_frame_id: int) -> None:
        for obj in self.map.get_all():
            if not obj.observations:
                continue
            last_seen = max(obs.frame_id for obs in obj.observations)
            if current_frame_id - last_seen > 30:
                obj.visibility_status = VisibilityStatus.INACTIVE

    def _class_compatible(self, name_a: str, name_b: str) -> bool:
        if name_a == name_b:
            return True
        return name_b in self._class_compat.get(name_a, [])

    def _class_cost(self, name_a: str, name_b: str) -> float:
        if name_a == name_b:
            return 0.0
        if name_b in self._class_compat.get(name_a, []):
            return 0.5
        return 1e9
