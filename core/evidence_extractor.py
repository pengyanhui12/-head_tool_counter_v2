"""证据帧提取器 — 选择最佳视角帧，回读视频，绘制标注，保存 JPG"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.types import GlobalObject, GlobalDetection


class EvidenceExtractor:
    def __init__(self, video_path: str | None = None):
        self.video_path = video_path

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
            gd = self.select_best(obj)
            if gd is None:
                continue

            if obj.persistent_id is None:
                continue

            # Read frame from video
            frame = self._read_frame(cap, gd.frame_id, frame_cache)
            if frame is None:
                continue

            # Draw bbox, ID, class
            annotated = self._draw_annotation(frame, obj, gd)

            filename = f"{obj.persistent_id}_{obj.class_name}.jpg"
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
        x1, y1, x2, y2 = 0, 0, 0, 0
        if obj.observations:
            obs = obj.observations[0]
            # Try to get bbox from RawDetection via track_id; not stored in GlobalDetection.
            # Fall back to projected_corners for rough bbox in pixel space.
            corners = obs.projected_corners
            x1 = int(corners[:, 0].min())
            y1 = int(corners[:, 1].min())
            x2 = int(corners[:, 0].max())
            y2 = int(corners[:, 1].max())

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{obj.persistent_id}: {obj.class_name}"
        cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        conf_label = f"conf={obj.confidence:.2f} status={obj.confirmation_status.value}"
        cv2.putText(img, conf_label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return img
