"""简化检测跟踪器 — IoU + 中心距离 Hungarian 匹配 + inactive 重激活

不称为 ByteTrack 或 BoT-SORT。
preview() 不修改状态，用于在关键帧判定之前收集触发信号。
update() 每帧最多调用一次。

不变量:
- T1: 一次 update 中，一个 track 最多匹配一个 detection
- T2: 一个 detection 最多分配给一个 track
- T3: active、inactive、lost 三个集合互斥
- T4: lost 不参与普通重激活
- T5: 未运行检测不等于检测结果为空
- T6: missed 按真实 frame_id 差推进
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
        inactive_min_iou: float = 0.30,
        inactive_max_center_distance_ratio: float = 0.12,
        quality_drop_trigger_ratio: float = 0.70,
        quality_drop_rearm_ratio: float = 0.85,
        quality_drop_min_history: int = 5,
    ):
        self.max_missed = max_missed_detection_frames
        self.lost_reactivation = lost_reactivation_frames
        self.min_iou = min_iou
        self.max_center_dist_ratio = max_center_distance_ratio
        self.iou_weight = iou_weight
        self.center_weight = center_weight
        self._class_compat = class_compatibility or {}
        self.inactive_min_iou = inactive_min_iou
        self.inactive_max_center_dist = inactive_max_center_distance_ratio

        # quality_drop 边沿触发参数
        self.quality_drop_trigger_ratio = quality_drop_trigger_ratio
        self.quality_drop_rearm_ratio = quality_drop_rearm_ratio
        self.quality_drop_min_history = quality_drop_min_history

        self._tracks: dict[int, Track] = {}
        self._next_track_id = 0
        self._image_w: int = 640
        self._image_h: int = 480
        self._current_frame_id: int = 0

        # quality_drop 边沿触发状态
        self._track_in_drop: dict[int, bool] = {}
        self._last_l2_frame: int = -999
        self._quality_drop_armed: bool = True

    # ── public API ──

    @staticmethod
    def no_detection_run() -> TrackerPreview:
        """L2 未运行时使用的预览——不产生任何触发信号。"""
        return TrackerPreview(
            unmatched_detection_indices=[],
            l2_new_unmatched_detection=False,
            track_quality_drop=False,
        )

    def preview(
        self, detections: list[DetectionCandidate], l2_was_run: bool = True
    ) -> TrackerPreview:
        """获取 tracker 预览信号，不修改状态。

        Args:
            detections: L2 检测结果（空列表仅当 L2 运行但无检测时才有意义）
            l2_was_run: 本帧是否实际运行了 L2 检测
        """
        if not l2_was_run:
            return self.no_detection_run()

        unmatched_indices = self._unmatched_indices(detections)
        l2_new = len(unmatched_indices) > 0

        # 边沿触发的 quality_drop
        quality_drop = False
        for t in self._tracks.values():
            if t.state == "active" and self.track_quality_dropped(t.track_id):
                quality_drop = True
                break

        return TrackerPreview(
            unmatched_detection_indices=unmatched_indices,
            l2_new_unmatched_detection=l2_new,
            track_quality_drop=quality_drop,
        )

    def advance_frame(self, frame_id: int) -> None:
        """推进时间但不运行检测——用于质量不合格帧。
        只更新 last_update_frame_id，不增加 missed_frames。
        """
        self._current_frame_id = frame_id
        for t in self._tracks.values():
            t.last_update_frame_id = frame_id

    def update(
        self, detections: list[DetectionCandidate], frame_id: int | None = None
    ) -> list[TrackedDetection]:
        """用检测更新跟踪器。

        严格区分 active/inactive/lost:
        - active: 参与匹配 + 创建新 track
        - inactive: 可重激活（更高的门控）
        - lost: 不参与任何匹配，不清除
        """
        if frame_id is not None:
            self._current_frame_id = frame_id
        else:
            self._current_frame_id += 1

        if detections:
            self._image_w = detections[0].image_width
            self._image_h = detections[0].image_height

        active_tracks = {tid: t for tid, t in self._tracks.items() if t.state == "active"}
        inactive_tracks = {tid: t for tid, t in self._tracks.items() if t.state == "inactive"}

        results: list[TrackedDetection] = []
        matched_det_indices: set[int] = set()
        matched_track_ids: set[int] = set()
        diag = np.sqrt(self._image_w**2 + self._image_h**2)

        # Phase 1: match active tracks to detections
        if active_tracks and detections:
            track_ids = list(active_tracks.keys())
            cost = np.full((len(track_ids), len(detections)), 1e9)

            for ti, tid in enumerate(track_ids):
                t = active_tracks[tid]
                for di, det in enumerate(detections):
                    c_cls = self._class_cost(t, det)
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
                # 确保 track_id 和 detection index 唯一
                if tid in matched_track_ids:
                    continue
                if c in matched_det_indices:
                    continue
                matched_det_indices.add(c)
                matched_track_ids.add(tid)
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
                    c_cls = self._class_cost(t, det)
                    if c_cls >= 1e9:
                        continue
                    iou = self._compute_iou(t.bbox, det.bbox)
                    center_dist = self._center_distance(t.bbox, det.bbox) / diag
                    # inactive 使用更严格的门控
                    if iou >= self.inactive_min_iou and center_dist <= self.inactive_max_center_dist:
                        react_cost[ti, dj] = (
                            self.iou_weight * (1.0 - iou)
                            + self.center_weight * center_dist
                            + 0.01 * c_cls
                        )

            ri2, ci2 = linear_sum_assignment(react_cost)
            for r, c in zip(ri2, ci2):
                if react_cost[r, c] >= 1e9:
                    continue
                tid = inactive_ids[r]
                det_idx, det = remaining_det[c]
                if tid in matched_track_ids:
                    continue
                if det_idx in matched_det_indices:
                    continue
                t = self._tracks[tid]
                t.state = "active"
                t.missed_frames = 0
                t.bbox = det.bbox
                t.confidence = det.confidence
                t.class_id = det.class_id
                t.class_name = det.class_name
                t.age += 1
                t.last_detection_frame_id = self._current_frame_id
                t.last_update_frame_id = self._current_frame_id
                t.confidence_history.append(det.confidence)
                t.detection_history.append(det)
                results.append(TrackedDetection(
                    candidate=det, track_id=tid,
                    track_age=t.age, is_new_track=False,
                ))
                matched_det_indices.add(det_idx)
                matched_track_ids.add(tid)

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
                last_update_frame_id=self._current_frame_id,
                last_detection_frame_id=self._current_frame_id,
                generation=0,
            )
            results.append(TrackedDetection(
                candidate=det, track_id=tid,
                track_age=1, is_new_track=True,
            ))

        # Mark unmatched active tracks as missed
        for tid in list(active_tracks.keys()):
            if tid in matched_track_ids:
                continue
            t = self._tracks[tid]
            t.missed_frames += 1
            t.last_update_frame_id = self._current_frame_id
            if t.missed_frames > self.lost_reactivation:
                t.state = "lost"
            elif t.missed_frames > self.max_missed:
                t.state = "inactive"

        # Advance inactive tracks that didn't get reactivated → towards lost
        for tid in list(inactive_tracks.keys()):
            if tid in matched_track_ids:
                continue
            t = self._tracks[tid]
            t.missed_frames += 1
            t.last_update_frame_id = self._current_frame_id
            if t.missed_frames > self.lost_reactivation:
                t.state = "lost"

        # 验证不变量: track_id 和 detection_index 唯一
        result_tids = {td.track_id for td in results}
        assert len(result_tids) == len(results), (
            f"Tracker: duplicate track_id in results"
        )

        return results

    def get_active_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.state == "active"]

    def get_inactive_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.state == "inactive"]

    def track_quality_dropped(self, track_id: int) -> bool:
        """边沿触发质量下降检测。

        - 仅在 L2 运行且 track 得到新检测时检查
        - 从正常降到低于 trigger_ratio 时触发一次
        - 连续下降期间不重复触发
        - 恢复到 rearm_ratio 以上后才能再次触发
        """
        t = self._tracks.get(track_id)
        if t is None or len(t.confidence_history) < self.quality_drop_min_history:
            return False

        recent = np.mean(t.confidence_history[-3:])
        full_mean = np.mean(t.confidence_history)
        if full_mean <= 0:
            return False

        ratio = recent / full_mean
        was_in_drop = self._track_in_drop.get(track_id, False)

        if not was_in_drop and ratio < self.quality_drop_trigger_ratio:
            # 进入下降状态
            self._track_in_drop[track_id] = True
            return True

        if was_in_drop and ratio >= self.quality_drop_rearm_ratio:
            # 恢复，重新 arm
            self._track_in_drop[track_id] = False

        return False

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
        t.last_detection_frame_id = self._current_frame_id
        t.last_update_frame_id = self._current_frame_id
        t.confidence_history.append(det.confidence)
        t.detection_history.append(det)

    def _unmatched_indices(self, detections: list[DetectionCandidate]) -> list[int]:
        active = {tid: t for tid, t in self._tracks.items() if t.state == "active"}
        unmatched = []
        for di, det in enumerate(detections):
            matched = False
            for tid, t in active.items():
                if not self._class_compatible_track(t, det):
                    continue
                iou = self._compute_iou(t.bbox, det.bbox)
                if iou >= self.min_iou:
                    matched = True
                    break
            if not matched:
                unmatched.append(di)
        return unmatched

    def _class_cost(self, track: Track, det: DetectionCandidate) -> float:
        """类别代价：同类别=0，兼容=0.5，不兼容=inf。"""
        if track.class_name == det.class_name:
            return 0.0
        compat_list = self._class_compat.get(track.class_name, [])
        if det.class_name in compat_list:
            return 0.5
        return float("inf")

    def _class_compatible_track(self, track: Track, det: DetectionCandidate) -> bool:
        if track.class_name == det.class_name:
            return True
        return det.class_name in self._class_compat.get(track.class_name, [])

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
