"""简化检测跟踪器 — IoU + 中心距离 Hungarian 匹配 + inactive 重激活

不称为 ByteTrack 或 BoT-SORT。
preview() 不修改状态，用于在关键帧判定之前收集触发信号。
update() 每帧最多调用一次。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from core.types import (
    DetectionCandidate,
    TrackedDetection,
    TrackerPreview,
    Track,
)


class SimpleDetectionTracker:
    def __init__(
        self,
        max_missed_detection_frames: int = 5,
        lost_reactivation_frames: int = 10,
        min_iou: float = 0.20,
        max_center_distance_ratio: float = 0.20,
        iou_weight: float = 0.60,
        center_weight: float = 0.40,
        class_compatibility: dict[str, list[str]] | None = None,
    ):
        self.max_missed = max_missed_detection_frames
        self.lost_reactivation = lost_reactivation_frames
        self.min_iou = min_iou
        self.max_center_dist_ratio = max_center_distance_ratio
        self.iou_weight = iou_weight
        self.center_weight = center_weight
        self._class_compat = class_compatibility or {}
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 0
        self._image_w: int = 640
        self._image_h: int = 480

    # ── public API ──

    def preview(self, detections: list[DetectionCandidate]) -> TrackerPreview:
        unmatched_indices = self._unmatched_indices(detections)
        l2_new = len(unmatched_indices) > 0
        quality_drop = any(
            self.track_quality_dropped(t.track_id)
            for t in self._tracks.values()
            if t.state == "active"
        )
        return TrackerPreview(
            unmatched_detection_indices=unmatched_indices,
            l2_new_unmatched_detection=l2_new,
            track_quality_drop=quality_drop,
        )

    def update(self, detections: list[DetectionCandidate]) -> list[TrackedDetection]:
        if detections:
            self._image_w = detections[0].image_width
            self._image_h = detections[0].image_height

        active_tracks = {tid: t for tid, t in self._tracks.items() if t.state != "lost"}
        inactive_tracks = {tid: t for tid, t in self._tracks.items()
                           if t.state == "inactive"}
        results: list[TrackedDetection] = []
        matched_det_indices: set[int] = set()
        diag = np.sqrt(self._image_w**2 + self._image_h**2)

        # Phase 1: match active tracks to detections
        if active_tracks and detections:
            track_ids = list(active_tracks.keys())
            cost = np.full((len(track_ids), len(detections)), 1e9)

            for ti, tid in enumerate(track_ids):
                t = active_tracks[tid]
                for di, det in enumerate(detections):
                    c_cls = self._class_cost(t.class_id, det.class_id)
                    if c_cls >= 1e9:
                        continue
                    iou = self._compute_iou(t.bbox, det.bbox)
                    center_dist = self._center_distance(t.bbox, det.bbox) / diag
                    cost[ti, di] = (
                        self.iou_weight * (1.0 - iou)
                        + self.center_weight * center_dist
                        + 0.01 * c_cls
                    )
                    if iou < self.min_iou or center_dist > self.max_center_dist_ratio:
                        cost[ti, di] = 1e9

            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] >= 1e9:
                    continue
                tid = track_ids[r]
                det = detections[c]
                matched_det_indices.add(c)
                self._update_track(tid, det)
                t = self._tracks[tid]
                results.append(TrackedDetection(
                    candidate=det, track_id=tid,
                    track_age=t.age, is_new_track=False,
                ))

        # Phase 2: reactivate inactive tracks for remaining detections
        remaining_det = [(di, det) for di, det in enumerate(detections)
                         if di not in matched_det_indices]
        if inactive_tracks and remaining_det:
            inactive_ids = list(inactive_tracks.keys())
            react_cost = np.full((len(inactive_ids), len(remaining_det)), 1e9)
            for ti, tid in enumerate(inactive_ids):
                t = inactive_tracks[tid]
                for dj, (_, det) in enumerate(remaining_det):
                    if t.class_id != det.class_id:
                        continue
                    iou = self._compute_iou(t.bbox, det.bbox)
                    center_dist = self._center_distance(t.bbox, det.bbox) / diag
                    if iou >= self.min_iou and center_dist <= self.max_center_dist_ratio:
                        react_cost[ti, dj] = (
                            self.iou_weight * (1.0 - iou)
                            + self.center_weight * center_dist
                        )

            ri2, ci2 = linear_sum_assignment(react_cost)
            for r, c in zip(ri2, ci2):
                if react_cost[r, c] >= 1e9:
                    continue
                tid = inactive_ids[r]
                det_idx, det = remaining_det[c]
                # reuse old track_id
                t = self._tracks[tid]
                t.state = "active"
                t.missed_frames = 0
                t.bbox = det.bbox
                t.confidence = det.confidence
                t.class_id = det.class_id
                t.class_name = det.class_name
                t.age += 1
                t.confidence_history.append(det.confidence)
                t.detection_history.append(det)
                results.append(TrackedDetection(
                    candidate=det, track_id=tid,
                    track_age=t.age, is_new_track=False,
                ))
                matched_det_indices.add(det_idx)

        # Phase 3: create new tracks for still-unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                age=1,
                missed_frames=0,
                state="active",
                confidence_history=[det.confidence],
                detection_history=[det],
            )
            results.append(TrackedDetection(
                candidate=det, track_id=tid,
                track_age=1, is_new_track=True,
            ))

        # Mark unmatched active tracks as missed
        matched_tids = {td.track_id for td in results}
        for tid in list(active_tracks.keys()):
            if tid in matched_tids:
                continue
            t = self._tracks[tid]
            t.missed_frames += 1
            if t.missed_frames > self.lost_reactivation:
                t.state = "lost"
            elif t.missed_frames > self.max_missed:
                t.state = "inactive"

        return results

    def get_active_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.state == "active"]

    def track_quality_dropped(self, track_id: int) -> bool:
        t = self._tracks.get(track_id)
        if t is None or len(t.confidence_history) < 3:
            return False
        recent = np.mean(t.confidence_history[-3:])
        full = np.mean(t.confidence_history)
        return full > 0 and recent / full < 0.7

    # ── internal ──

    def _update_track(self, tid: int, det: DetectionCandidate) -> None:
        t = self._tracks[tid]
        t.bbox = det.bbox
        t.confidence = det.confidence
        t.class_id = det.class_id
        t.class_name = det.class_name
        t.age += 1
        t.missed_frames = 0
        t.state = "active"
        t.confidence_history.append(det.confidence)
        t.detection_history.append(det)

    def _unmatched_indices(self, detections: list[DetectionCandidate]) -> list[int]:
        active = {tid: t for tid, t in self._tracks.items() if t.state == "active"}
        unmatched = []
        for di, det in enumerate(detections):
            matched = False
            for tid, t in active.items():
                if t.class_id != det.class_id:
                    continue
                iou = self._compute_iou(t.bbox, det.bbox)
                if iou >= self.min_iou:
                    matched = True
                    break
            if not matched:
                unmatched.append(di)
        return unmatched

    def _class_cost(self, cid_a: int, cid_b: int) -> float:
        if cid_a == cid_b:
            return 0.0
        a_name = self._tracks.get(cid_a)
        b_name = self._tracks.get(cid_b)
        if a_name is not None and b_name is not None:
            if a_name.class_name == b_name.class_name:
                return 0.0
            compat = self._class_compat.get(a_name.class_name, [])
            if b_name.class_name in compat:
                return 0.5
        return 1e9

    @staticmethod
    def _compute_iou(a: tuple, b: tuple) -> float:
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _center_distance(a: tuple, b: tuple) -> float:
        ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        return float(np.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2))
