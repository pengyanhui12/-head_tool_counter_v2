"""证据帧提取器 — 选择最佳视角帧，回读视频，绘制标注，保存 JPG"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.types import GlobalObject, GlobalDetection


class EvidenceExtractor:
    def __init__(self, video_path: str | None = None, track_cache: dict | None = None):
        """track_cache: {track_id: DetectionCandidate} 从 tracker 获取原始像素 bbox。"""
        self.video_path = video_path
        self._track_cache = track_cache or {}

    @staticmethod
    def select_best(obj: GlobalObject) -> GlobalDetection | None:
        if not obj.observations:
            return None
        return max(
            obj.observations,
            key=lambda o: o.sharpness * o.detection_confidence,
        )

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

        for obj in objects:
            # 跳过 REJECTED 对象（已被合并到其他对象）
            if obj.confirmation_status.value == "rejected":
                continue

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
