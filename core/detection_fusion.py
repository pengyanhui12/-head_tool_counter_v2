"""检测融合器：同类别 NMS，以及受重叠约束的中心距离去重。"""

import numpy as np

from core.types import DetectionCandidate


class DetectionFusion:
    def __init__(
        self,
        iou_threshold: float = 0.65,
        center_merge_distance_px: float = 40.0,
        center_merge_min_ios: float = 0.30,
        per_class_center_merge_distances: dict[str, float] | None = None,
    ):
        self.iou_threshold = iou_threshold
        self.center_merge_distance_px = center_merge_distance_px
        self.center_merge_min_ios = center_merge_min_ios
        self.per_class_center_merge_distances = (
            per_class_center_merge_distances or {}
        )

    def fuse(
        self,
        l1: list[DetectionCandidate],
        l3: list[DetectionCandidate],
    ) -> list[DetectionCandidate]:
        """融合 L1/L3，并保证只有 L1 时也执行同类 NMS。"""
        by_class: dict[int, list[DetectionCandidate]] = {}
        for detection in l1 + l3:
            by_class.setdefault(detection.class_id, []).append(detection)

        nms_result: list[DetectionCandidate] = []
        for detections in by_class.values():
            boxes = np.array(
                [detection.bbox for detection in detections],
                dtype=np.float32,
            )
            scores = np.array(
                [detection.confidence for detection in detections],
                dtype=np.float32,
            )
            keep = self._nms(boxes, scores, self.iou_threshold)
            nms_result.extend(detections[index] for index in keep)

        if len(nms_result) <= 1:
            return nms_result
        return self._deduplicate_by_center_distance(nms_result)

    def _deduplicate_by_center_distance(
        self,
        detections: list[DetectionCandidate],
    ) -> list[DetectionCandidate]:
        """抑制中心接近且明显重叠的同类框，保留置信度最高者。"""
        by_class: dict[int, list[DetectionCandidate]] = {}
        for detection in detections:
            by_class.setdefault(detection.class_id, []).append(detection)

        merged: list[DetectionCandidate] = []
        for class_detections in by_class.values():
            sorted_detections = sorted(
                class_detections,
                key=lambda detection: detection.confidence,
                reverse=True,
            )
            distance_threshold = (
                self.per_class_center_merge_distances.get(
                    sorted_detections[0].class_name,
                    self.center_merge_distance_px,
                )
            )
            suppressed: set[int] = set()

            for index, primary in enumerate(sorted_detections):
                if index in suppressed:
                    continue
                merged.append(primary)
                if distance_threshold <= 0:
                    continue

                primary_center = self._bbox_center(primary.bbox)
                for other_index in range(index + 1, len(sorted_detections)):
                    if other_index in suppressed:
                        continue
                    secondary = sorted_detections[other_index]
                    secondary_center = self._bbox_center(secondary.bbox)
                    center_distance = float(np.hypot(
                        primary_center[0] - secondary_center[0],
                        primary_center[1] - secondary_center[1],
                    ))
                    overlap = self._intersection_over_smaller(
                        primary.bbox,
                        secondary.bbox,
                    )
                    if (
                        center_distance < distance_threshold
                        and overlap >= self.center_merge_min_ios
                    ):
                        suppressed.add(other_index)

        return merged

    @staticmethod
    def _bbox_center(bbox: tuple) -> tuple[float, float]:
        return (
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
        )

    @staticmethod
    def _intersection_over_smaller(a: tuple, b: tuple) -> float:
        intersection_width = max(
            0.0,
            min(a[2], b[2]) - max(a[0], b[0]),
        )
        intersection_height = max(
            0.0,
            min(a[3], b[3]) - max(a[1], b[1]),
        )
        intersection = intersection_width * intersection_height
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        smaller_area = min(area_a, area_b)
        return intersection / smaller_area if smaller_area > 0 else 0.0

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while len(order) > 0:
            index = order[0]
            keep.append(int(index))

            intersection_x1 = np.maximum(x1[index], x1[order[1:]])
            intersection_y1 = np.maximum(y1[index], y1[order[1:]])
            intersection_x2 = np.minimum(x2[index], x2[order[1:]])
            intersection_y2 = np.minimum(y2[index], y2[order[1:]])

            intersection_width = np.maximum(
                0.0,
                intersection_x2 - intersection_x1,
            )
            intersection_height = np.maximum(
                0.0,
                intersection_y2 - intersection_y1,
            )
            intersection = intersection_width * intersection_height
            union = areas[index] + areas[order[1:]] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep
