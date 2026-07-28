"""对象关联器 — 门控 + 匈牙利匹配 + 多帧投票 + 后处理合并 + DBSCAN 复核"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from core.types import (
    GlobalDetection,
    GlobalObject,
    ConfirmationStatus,
    VisibilityStatus,
    ReviewFlag,
    RebuildResult,
)
from core.global_object_map import GlobalObjectMap


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

        self.map = GlobalObjectMap()
        self._all_gd: list[GlobalDetection] = []
        self._track_vote_counts: dict[int, int] = {}
        self._track_to_object: dict[int, str] = {}
        # 在线共现记录
        self._frame_co_occurred: set[tuple[int, str, str]] = set()
        self._co_occurred_pairs: set[frozenset[str]] = set()

    # ── 在线帧处理 ──

    def ingest_frame(
        self, frame_id: int, global_detections: list[GlobalDetection]
    ) -> list[str]:
        """处理一帧的全局检测，关联到已有对象或创建新对象。

        匹配优先级：
        1. track_id 强关联
        2. 帧级全局匈牙利（空间 + 类别 + 尺寸）
        3. 创建新对象
        """
        affected: list[str] = []
        if not global_detections:
            return affected

        unmatched_gds: list[GlobalDetection] = []
        for gd in global_detections:
            self._all_gd.append(gd)
            existing_pid = self._track_to_object.get(gd.track_id)
            if existing_pid is not None:
                obj = self.map.get_by_provisional(existing_pid)
                if obj is not None and self._class_compatible(obj.class_name, gd.class_name):
                    self._update_object(obj, gd)
                    affected.append(obj.provisional_id)
                    continue
            unmatched_gds.append(gd)

        if not unmatched_gds:
            self._prune(frame_id)
            return affected

        all_objects = self.map.get_all()
        if not all_objects:
            for gd in unmatched_gds:
                obj = self.map.create_object(gd)
                self._track_to_object[gd.track_id] = obj.provisional_id
                affected.append(obj.provisional_id)
            self._prune(frame_id)
            return affected

        # 帧级全局匈牙利
        n_det = len(unmatched_gds)
        n_obj = len(all_objects)
        global_cost = np.full((n_det, n_obj), 1e9)

        for di, gd in enumerate(unmatched_gds):
            for oi, obj in enumerate(all_objects):
                if not self._class_compatible(obj.class_name, gd.class_name):
                    continue
                pos_dist = np.linalg.norm(
                    np.array(gd.polygon_centroid) - np.array(obj.centroid_xy)
                )
                # 在线阶段使用更保守的门控（40%），避免贪心合并独立实例
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
        matched_det: set[int] = set()
        for r, c in zip(ri, ci):
            if global_cost[r, c] >= 1e9 or global_cost[r, c] > self.max_cost:
                continue
            self._update_object(all_objects[c], unmatched_gds[r])
            affected.append(all_objects[c].provisional_id)
            matched_det.add(r)

        for di in range(n_det):
            if di not in matched_det:
                gd = unmatched_gds[di]
                obj = self.map.create_object(gd)
                self._track_to_object[gd.track_id] = obj.provisional_id
                affected.append(obj.provisional_id)

        # 记录共现 — 仅对空间足够分离的检测（避免 YOLO 重复检测误标）
        self._record_co_occurrences(frame_id, global_detections)

        self._prune(frame_id)
        return affected

    def _record_co_occurrences(
        self, frame_id: int, global_detections: list[GlobalDetection]
    ) -> None:
        """记录本帧内不同对象间的共现关系。

        用像素空间 IoU 判定 YOLO 重复检测 vs 独立实例：
        - IoU=0: 两个框完全不重叠 → 独立的两个工具 → 记录共现
        - IoU>0: 两个框有重叠 → 同一工具的重复检测 → 不记录共现

        只对同类别对象记录共现（不同类别的工具天然独立）。
        """
        # 用 track_id 查找 pid（避免 gd in obj.observations 的 numpy 比较问题）
        pid_per_det: list[tuple[GlobalDetection, str | None]] = []
        for gd in global_detections:
            pid = self._track_to_object.get(gd.track_id)
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
                # 同名同类：IoU==0 → 独立实例
                if gd_a.bbox_pixels and gd_b.bbox_pixels:
                    iou = self._compute_bbox_iou(gd_a.bbox_pixels, gd_b.bbox_pixels)
                    if iou == 0.0:
                        self._frame_co_occurred.add((frame_id, pid_a, pid_b))
                        self._co_occurred_pairs.add(pair)

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

    # ── 后处理合并（final_review）──

    def final_review(self) -> None:
        """后处理：track_id 硬合并 + DBSCAN 标记疑似重复。

        1. track_id 重叠 → 硬合并（同一 tracker 实例的确定性证据）
        2. DBSCAN 空间聚类 → 标记 LIKELY_DUPLICATE 供人工复核
        """
        self._merge_non_co_occurred()
        self._mark_likely_duplicates()

    def _merge_non_co_occurred(self) -> None:
        """track_id 重叠 + centroid 硬合并。

        track_id 重叠是唯一确定性证据（同一 tracker 实例 → 同一物理工具）。
        此外 centroid < 30px 且不共现 → 也合并（精度范围内的重投影误差）。
        不基于 centroid 距离对其他情况做合并——同类工具可能紧邻放置。
        """
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
                    # 硬合并条件1：track_id 重叠
                    if primary.track_ids & secondary.track_ids:
                        self._merge_objects(primary, secondary)
                        merged.add(secondary.provisional_id)
                        self._reevaluate(primary)
                        continue
                    # 硬合并条件2：centroid < 30px 且不共现 → 精度范围内的重投影误差
                    pair = frozenset([primary.provisional_id, secondary.provisional_id])
                    if pair in self._co_occurred_pairs:
                        continue
                    dist = np.linalg.norm(
                        np.array(primary.centroid_xy) - np.array(secondary.centroid_xy)
                    )
                    if dist < 30.0:
                        self._merge_objects(primary, secondary)
                        merged.add(secondary.provisional_id)
                        self._reevaluate(primary)

    def _merge_complementary_coverage(self) -> None:
        """互补覆盖合并：keyframe 集合完全不重叠 + 从未共现 + centroid 较近。"""
        objs = [o for o in self.map.get_all()
                if o.confirmation_status == ConfirmationStatus.CONFIRMED]
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
                    if primary.keyframe_ids & secondary.keyframe_ids:
                        continue  # 有时间重叠，不是互补
                    pair = frozenset([primary.provisional_id, secondary.provisional_id])
                    if pair in self._co_occurred_pairs:
                        continue
                    dist = np.linalg.norm(
                        np.array(primary.centroid_xy) - np.array(secondary.centroid_xy)
                    )
                    if dist < self.max_position_distance * 0.6:
                        self._merge_objects(primary, secondary)
                        merged.add(secondary.provisional_id)
                        self._reevaluate(primary)

    def _mark_likely_duplicates(self) -> None:
        """DBSCAN 残差标记。"""
        class_objects: dict[str, list[GlobalObject]] = {}
        for obj in self.map.get_all():
            if obj.confirmation_status == ConfirmationStatus.CONFIRMED:
                class_objects.setdefault(obj.class_name, []).append(obj)

        for _class_name, objs in class_objects.items():
            if len(objs) < 2:
                continue
            pts = np.array([obj.centroid_xy for obj in objs])
            clustering = DBSCAN(eps=self.max_position_distance * 0.5, min_samples=2).fit(pts)
            labels = clustering.labels_

            cluster_map: dict[int, list[GlobalObject]] = {}
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                cluster_map.setdefault(label, []).append(objs[i])

            for cluster_objs in cluster_map.values():
                unique_ids = {o.provisional_id for o in cluster_objs}
                if len(unique_ids) > 1:
                    for o in cluster_objs:
                        o.review_flags.add(ReviewFlag.LIKELY_DUPLICATE)

    def _merge_objects(self, primary: GlobalObject, secondary: GlobalObject) -> None:
        """将 secondary 合并到 primary，secondary 标 REJECTED。"""
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
        for tid in secondary.track_ids:
            self._track_to_object[tid] = primary.provisional_id
        secondary.confirmation_status = ConfirmationStatus.REJECTED
        secondary.uncertainty_reasons.append(f"merged_into_{primary.provisional_id}")

    # ── 内部 helper ──

    def _build_cost_matrix(
        self, gd: GlobalDetection, candidates: list[GlobalObject]
    ) -> np.ndarray:
        n = len(candidates)
        cost = np.full((1, n), 1e9)
        for j, obj in enumerate(candidates):
            pos_cost = np.linalg.norm(
                np.array(gd.polygon_centroid) - np.array(obj.centroid_xy)
            )
            area_min, area_max = obj.area_range
            size_cost = (abs(gd.polygon_area - (area_min + area_max) / 2) / area_max
                         if area_max > 0 else 0.0)
            class_cost = self._class_cost(obj.class_name, gd.class_name)
            if class_cost >= 1e9:
                continue
            total = (
                self.w_pos * pos_cost / self.max_position_distance
                + self.w_size * min(size_cost, 1.0)
                + self.w_class * class_cost
            )
            if pos_cost <= self.max_position_distance:
                cost[0, j] = total
        return cost

    def _apply_hard_constraints(
        self, gd: GlobalDetection, candidates: list[GlobalObject],
        cost: np.ndarray, frame_detections: list[GlobalDetection]
    ) -> np.ndarray:
        return cost

    def _update_object(self, obj: GlobalObject, gd: GlobalDetection) -> None:
        obj.observations.append(gd)
        obj.observation_count += 1
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

        tid = gd.track_id
        votes_used = self._track_vote_counts.get(tid, 0)
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
            self._track_vote_counts[tid] = votes_used + 1

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
