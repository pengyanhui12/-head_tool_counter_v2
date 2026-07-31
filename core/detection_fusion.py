"""检测融合器 — L1 + L3 同类别 NMS + 同帧近距离去重

关键修复：YOLO 在同一帧内可能对同一物理工具产生多个检测框。
这些框的 IoU 可能 < 0.65（如一个框覆盖工具头部，另一个覆盖工具尾部），
但它们属于同一工具。需要通过中心点距离来合并这些"近重复"检测。

规则：
1. 标准 NMS (IoU > iou_threshold) — 消除明显的重叠框
2. 中心点距离去重 (同类别，中心点距离 < center_merge_distance_px) — 合并散开的重复框，保留置信度最高的
"""
import numpy as np

from core.types import DetectionCandidate


class DetectionFusion:
    def __init__(self, iou_threshold: float = 0.65, center_merge_distance_px: float = 40.0):
        self.iou_threshold = iou_threshold
        self.center_merge_distance_px = center_merge_distance_px

    def fuse(
        self,
        l1: list[DetectionCandidate],
        l3: list[DetectionCandidate],
    ) -> list[DetectionCandidate]:
        if not l3:
            result = list(l1)
        elif not l1:
            result = list(l3)
        else:
            # Group by class_id
            by_class: dict[int, list[DetectionCandidate]] = {}
            for det in l1 + l3:
                by_class.setdefault(det.class_id, []).append(det)

            result: list[DetectionCandidate] = []
            for dets in by_class.values():
                boxes = np.array([d.bbox for d in dets], dtype=np.float32)
                scores = np.array([d.confidence for d in dets], dtype=np.float32)
                keep = self._nms(boxes, scores, self.iou_threshold)
                for i in keep:
                    result.append(dets[i])

        # ── 同类别中心点距离去重 ──
        if self.center_merge_distance_px > 0 and len(result) > 1:
            result = self._deduplicate_by_center_distance(result)

        return result

    def _deduplicate_by_center_distance(
        self, detections: list[DetectionCandidate]
    ) -> list[DetectionCandidate]:
        """合并同类别中中心点距离过近的检测，保留置信度最高的。"""
        by_class: dict[int, list[DetectionCandidate]] = {}
        for det in detections:
            by_class.setdefault(det.class_id, []).append(det)

        merged: list[DetectionCandidate] = []
        for _class_id, dets in by_class.items():
            if len(dets) <= 1:
                merged.extend(dets)
                continue

            # Sort by confidence descending
            dets_sorted = sorted(dets, key=lambda d: d.confidence, reverse=True)
            kept = []
            suppressed = set()

            for i, det_a in enumerate(dets_sorted):
                if i in suppressed:
                    continue
                # Merge all nearby dets into this one
                best_det = det_a
                for j, det_b in enumerate(dets_sorted):
                    if j <= i or j in suppressed:
                        continue
                    ca = ((det_a.bbox[0] + det_a.bbox[2]) / 2,
                          (det_a.bbox[1] + det_a.bbox[3]) / 2)
                    cb = ((det_b.bbox[0] + det_b.bbox[2]) / 2,
                          (det_b.bbox[1] + det_b.bbox[3]) / 2)
                    center_dist = np.sqrt((ca[0] - cb[0])**2 + (ca[1] - cb[1])**2)
                    if center_dist < self.center_merge_distance_px:
                        suppressed.add(j)
                        # Keep the one with higher confidence
                        if det_b.confidence > best_det.confidence:
                            best_det = det_b

                kept.append(best_det)

            merged.extend(kept)

        return merged

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]; y1 = boxes[:, 1]
        x2 = boxes[:, 2]; y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep
