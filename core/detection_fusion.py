"""检测融合器 — L1 + L3 同类别 NMS，禁止直接 extend"""
import numpy as np

from core.types import DetectionCandidate


class DetectionFusion:
    def __init__(self, iou_threshold: float = 0.65):
        self.iou_threshold = iou_threshold

    def fuse(
        self,
        l1: list[DetectionCandidate],
        l3: list[DetectionCandidate],
    ) -> list[DetectionCandidate]:
        if not l3:
            return list(l1)
        if not l1:
            return list(l3)

        # Group by class_id
        by_class: dict[int, list[DetectionCandidate]] = {}
        for det in l1 + l3:
            by_class.setdefault(det.class_id, []).append(det)

        fused: list[DetectionCandidate] = []
        for dets in by_class.values():
            boxes = np.array([d.bbox for d in dets], dtype=np.float32)
            scores = np.array([d.confidence for d in dets], dtype=np.float32)

            keep = self._nms(boxes, scores, self.iou_threshold)
            for i in keep:
                fused.append(dets[i])

        return fused

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
