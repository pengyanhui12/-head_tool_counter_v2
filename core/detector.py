"""YOLO 检测器 — L1/L2/L3 分层推理

所有层统一返回 list[DetectionCandidate]。
L2 仅用于触发和跟踪，不投影。
L1/L3 在 keyframe 上运行，融合后进入 RawDetection。
"""
import numpy as np

from core.types import DetectionCandidate


class Detector:
    def __init__(
        self,
        model_path: str = "models/best.pt",
        device: str = "cuda:0",
        l1_imgsz: int = 1280,
        l1_conf: float = 0.15,
        l1_iou: float = 0.65,
        l2_imgsz: int = 640,
        l2_conf: float = 0.10,
        l2_iou: float = 0.65,
        l3_imgsz: int = 1280,
        l3_conf: float = 0.10,
        l3_iou: float = 0.65,
    ):
        self.l1_imgsz = l1_imgsz
        self.l1_conf = l1_conf
        self.l1_iou = l1_iou
        self.l2_imgsz = l2_imgsz
        self.l2_conf = l2_conf
        self.l2_iou = l2_iou
        self.l3_imgsz = l3_imgsz
        self.l3_conf = l3_conf
        self.l3_iou = l3_iou

        self._model = None
        self._names: dict[int, str] = {}
        if model_path:
            self._load_model(model_path, device)

    def _load_model(self, path: str, device: str) -> None:
        from pathlib import Path
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. "
                f"Place your YOLO weights at models/best.pt"
            )
        from ultralytics import YOLO
        self._model = YOLO(path)
        self._names = self._model.names

    def detect(
        self,
        image: np.ndarray,
        level: str,
        frame_id: int,
        regions: list[tuple[int, int, int, int]] | None = None,
    ) -> list[DetectionCandidate]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        imgsz = getattr(self, f"{level.lower()}_imgsz")
        conf = getattr(self, f"{level.lower()}_conf")
        iou = getattr(self, f"{level.lower()}_iou")

        h, w = image.shape[:2]

        results = self._model(image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

        candidates: list[DetectionCandidate] = []
        for r in results:
            if r.boxes is None:
                continue
            for box, cls_id, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(float, box.tolist())
                cls_int = int(cls_id.item())
                candidates.append(
                    DetectionCandidate(
                        frame_id=frame_id,
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_int,
                        class_name=self._names.get(cls_int, "unknown"),
                        confidence=float(cf.item()),
                        source=level,
                        image_width=w,
                        image_height=h,
                    )
                )
        return candidates
