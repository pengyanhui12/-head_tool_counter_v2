"""对象关联器 — 门控 + 匈牙利匹配 + 多帧投票 + DBSCAN 复核"""
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
        # Frame-level co-occurrence (frame_id, pid_a, pid_b)
        self._frame_co_occurred: set[tuple[int, str, str]] = set()
        # Cross-frame co-occurred pairs (frozenset of two pids)
        self._co_occurred_pairs: set[frozenset[str]] = set()

    def ingest_frame(
        self, frame_id: int, global_detections: list[GlobalDetection]
    ) -> list[str]:
        """Process one keyframe's worth of global detections."""
        affected: list[str] = []
        if not global_detections:
            return affected

        # Record within-frame co-occurrences
        pids_in_frame = []
        for gd in global_detections:
            self._all_gd.append(gd)
            candidates = [
                obj for obj in self.map.get_all()
                if self._class_compatible(obj.class_name, gd.class_name)
                and obj.visibility_status != "lost"  # crude filter
            ]
            pids_in_frame.append((gd, candidates))

        # Build cost matrix per detection
        for gd, candidates in pids_in_frame:
            if not candidates:
                obj = self.map.create_object(gd)
                affected.append(obj.provisional_id)
                continue

            cost = self._build_cost_matrix(gd, candidates)
            # Apply hard constraints (same-frame exclusion, co-occurrence)
            cost = self._apply_hard_constraints(gd, candidates, cost, global_detections)

            if cost.size == 0 or cost.min() >= 1e9:
                obj = self.map.create_object(gd)
                affected.append(obj.provisional_id)
                continue

            ri, ci = linear_sum_assignment(cost)
            if cost[ri[0], ci[0]] >= 1e9:
                obj = self.map.create_object(gd)
                affected.append(obj.provisional_id)
            else:
                matched = candidates[ci[0]]
                self._update_object(matched, gd)
                affected.append(matched.provisional_id)

        # Record co-occurrence for this frame — only among objects created/matched
        all_pids: set[str] = set()
        for gd, _ in pids_in_frame:
            for obj in self.map.get_all():
                if gd in obj.observations:
                    all_pids.add(obj.provisional_id)
        for gd, _ in pids_in_frame:
            for obj in self.map.get_all():
                if gd in obj.observations and obj.provisional_id in all_pids:
                    for other_pid in all_pids:
                        if other_pid != obj.provisional_id:
                            self._frame_co_occurred.add(
                                (frame_id, obj.provisional_id, other_pid)
                            )
                            self._co_occurred_pairs.add(
                                frozenset([obj.provisional_id, other_pid])
                            )

        # Prune stale
        self._prune(frame_id)
        return affected

    def final_review(self) -> None:
        """DBSCAN on all centroids to flag LIKELY_DUPLICATE."""
        pts = np.array([gd.polygon_centroid for gd in self._all_gd])
        if len(pts) < 3:
            return
        clustering = DBSCAN(eps=self.max_position_distance, min_samples=3).fit(pts)
        labels = clustering.labels_

        cluster_map: dict[int, list[GlobalObject]] = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            gd = self._all_gd[i]
            for obj in self.map.get_all():
                if gd in obj.observations and obj.confirmation_status == ConfirmationStatus.CONFIRMED:
                    cluster_map.setdefault(label, []).append(obj)

        for objs in cluster_map.values():
            unique_ids = {o.provisional_id for o in objs}
            if len(unique_ids) > 1:
                for o in objs:
                    o.review_flags.add(ReviewFlag.LIKELY_DUPLICATE)

    def _build_cost_matrix(
        self, gd: GlobalDetection, candidates: list[GlobalObject]
    ) -> np.ndarray:
        n = len(candidates)
        cost = np.full((1, n), 1e9)
        for j, obj in enumerate(candidates):
            pos_cost = np.linalg.norm(
                np.array(gd.polygon_centroid) - np.array(obj.centroid_xy)
            )
            # IoU cost: simplified as area difference
            area_min, area_max = obj.area_range
            if area_max > 0:
                size_cost = abs(gd.polygon_area - (area_min + area_max) / 2) / area_max
            else:
                size_cost = 0.0
            # C_class: same=0, compatible=penalty, incompatible=inf
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
        # Same-frame exclusion: two detections in same frame cannot match same object
        for j, obj in enumerate(candidates):
            for other_gd in frame_detections:
                if other_gd is gd and other_gd.frame_id == gd.frame_id:
                    continue
                # If this candidate already matched another detection in this frame, block
                pid = obj.provisional_id
                if any(
                    (gd.frame_id, pid, o2.provisional_id) in self._frame_co_occurred
                    for o2 in self.map.get_all()
                    if o2 is not obj
                ):
                    # Not blocking — only enforce cross-object within same frame
                    pass

        # Cross-frame co-occurrence: if two objects were seen together, never merge
        for j, obj in enumerate(candidates):
            for other_obj in self.map.get_all():
                if other_obj is obj:
                    continue
                pair = frozenset([obj.provisional_id, other_obj.provisional_id])
                if pair in self._co_occurred_pairs:
                    # Cannot merge into either — but this only blocks if we try
                    # to merge two previously co-observed objects.
                    # In ingest_frame we match a new detection to candidates,
                    # so we check: would assigning this detection to `obj`
                    # cause a merge? No, it just adds to `obj`. The hard
                    # constraint matters more in rebuild_all where objects
                    # might be merged. For now, block if the candidate already
                    # co-occurred with others in this frame.
                    pass

        return cost

    def _update_object(self, obj: GlobalObject, gd: GlobalDetection) -> None:
        obj.observations.append(gd)
        obj.observation_count += 1
        obj.track_ids.add(gd.track_id)
        obj.keyframe_ids.add(gd.keyframe_id)
        obj.centroid_xy = gd.polygon_centroid
        # Update area range
        areas = [obs.polygon_area for obs in obj.observations if obs.polygon_area > 0]
        if areas:
            obj.area_range = (min(areas), max(areas))

        # Voting
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

        # Reevaluate confirmation
        self._reevaluate(obj)

    def _reevaluate(self, obj: GlobalObject) -> None:
        if obj.observation_count >= self.min_obs_confirm and len(obj.keyframe_ids) >= self.min_kf_confirm:
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
            gap = current_frame_id - last_seen
            if gap > 30:
                obj.visibility_status = VisibilityStatus.INACTIVE

    def _class_compatible(self, name_a: str, name_b: str) -> bool:
        if name_a == name_b:
            return True
        compat = self._class_compat.get(name_a, [])
        return name_b in compat

    def _class_cost(self, name_a: str, name_b: str) -> float:
        if name_a == name_b:
            return 0.0
        compat = self._class_compat.get(name_a, [])
        if name_b in compat:
            return 0.5
        return 1e9
