"""证据帧提取器 — 选择最佳视角帧，回读视频，绘制标注，保存 JPG"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.types import GlobalObject, GlobalDetection
from core.report_generator import get_display_objects


class EvidenceExtractor:
    def __init__(self, video_path: str | None = None, track_cache: dict | None = None):
        """track_cache: {track_id: DetectionCandidate} 从 tracker 获取原始像素 bbox。"""
        self.video_path = video_path
        self._track_cache = track_cache or {}

    @staticmethod
    def select_best(obj: GlobalObject) -> GlobalDetection | None:
        if not obj.observations:
            return None

        # 只有全局几何有效时才做空间一致性判断；否则保持原有回退行为。
        object_centroid = np.asarray(obj.centroid_xy, dtype=float)
        valid_observations = [
            observation
            for observation in obj.observations
            if np.all(np.isfinite(observation.polygon_centroid))
        ]
        if not np.all(np.isfinite(object_centroid)) or not valid_observations:
            return max(
                obj.observations,
                key=lambda observation: (
                    observation.sharpness
                    * observation.detection_confidence
                ),
            )

        # 用对象典型边长归一化距离，使不同尺寸工具采用一致的空间惩罚。
        valid_areas = [
            float(observation.polygon_area)
            for observation in valid_observations
            if np.isfinite(observation.polygon_area)
            and observation.polygon_area > 0
        ]
        spatial_scale = (
            float(np.sqrt(np.median(valid_areas)))
            if valid_areas
            else 1.0
        )
        spatial_scale = max(spatial_scale, 1.0)

        def evidence_score(observation: GlobalDetection) -> float:
            """综合图像质量、映射质量和全局位置一致性。"""
            observation_centroid = np.asarray(
                observation.polygon_centroid, dtype=float
            )
            normalized_distance = float(
                np.linalg.norm(observation_centroid - object_centroid)
                / spatial_scale
            )
            spatial_consistency = 1.0 / (1.0 + normalized_distance ** 2)
            mapping_quality = float(np.clip(
                observation.mapping_quality, 0.0, 1.0
            ))
            return (
                max(float(observation.sharpness), 0.0)
                * max(float(observation.detection_confidence), 0.0)
                * mapping_quality
                * spatial_consistency
            )

        return max(valid_observations, key=evidence_score)

    def extract(
        self,
        video_path: str,
        objects: list[GlobalObject],
        output_dir: str,
    ) -> dict[str, str]:
        cap = cv2.VideoCapture(video_path)
        frame_cache: dict[int, np.ndarray] = {}
        result: dict[str, str] = {}

        out = Path(output_dir) / "evidence"
        out.mkdir(parents=True, exist_ok=True)

        for obj in get_display_objects(objects):
            gd = self.select_best(obj)
            if gd is None:
                continue

            # Read frame from video
            frame = self._read_frame(cap, gd.frame_id, frame_cache)
            if frame is None:
                continue

            # Draw bbox using bbox_pixels from the best observation
            annotated = self._draw_annotation(frame, obj, gd)

            display_id = obj.persistent_id or obj.provisional_id
            filename = f"{display_id}_{obj.class_name}.jpg"
            filepath = out / filename
            cv2.imwrite(str(filepath), annotated)
            result[obj.provisional_id] = str(filepath)

        cap.release()
        return result

    @staticmethod
    def _read_frame(
        cap: cv2.VideoCapture,
        frame_id: int,
        cache: dict[int, np.ndarray],
    ) -> np.ndarray | None:
        if frame_id in cache:
            return cache[frame_id]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, img = cap.read()
        if ok:
            cache[frame_id] = img
        return img if ok else None

    @staticmethod
    def _draw_annotation(
        image: np.ndarray,
        obj: GlobalObject,
        gd: GlobalDetection,
    ) -> np.ndarray:
        img = image.copy()
        # GlobalDetection.bbox_pixels 已在 projector.project() 时保留原始像素 bbox
        bbox = getattr(gd, 'bbox_pixels', (0, 0, 0, 0))
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if x2 > x1 and y2 > y1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_id = obj.persistent_id or obj.provisional_id
            label = f"{display_id}: {obj.class_name}"
            cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            conf_label = f"conf={obj.confidence:.2f} status={obj.confirmation_status.value}"
            cv2.putText(img, conf_label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return img
