"""会话存储 — 保存/加载 pipeline 中间产物用于调试和复现"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


class SessionStore:
    def __init__(self, output_dir: str = "outputs"):
        self._base = Path(output_dir) / "sessions"
        self._session_dir: Path | None = None

    def create_session(self, video_path: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        self._session_dir = self._base / f"{video_name}_{timestamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        return str(self._session_dir)

    def save_config(self, config: dict, filename: str = "config.json") -> None:
        self._write_json(config, filename)

    def save_keyframes(self, keyframes: list[dict], filename: str = "keyframes.json") -> None:
        self._write_json(keyframes, filename)

    def save_objects(self, objects: list[dict], filename: str = "objects.json") -> None:
        self._write_json(self._serialize_objects(objects), filename)

    def save_global_detections(self, detections: list[dict], filename: str = "detections.json") -> None:
        self._write_json(detections, filename)

    def save_log(self, log_lines: list[str], filename: str = "pipeline.log") -> None:
        if self._session_dir is None:
            raise RuntimeError("No session created")
        (self._session_dir / filename).write_text("\n".join(log_lines))

    def _write_json(self, data, filename: str) -> None:
        if self._session_dir is None:
            raise RuntimeError("No session created")
        path = self._session_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str))

    @staticmethod
    def _serialize_objects(objects: list[dict]) -> list[dict]:
        result = []
        for obj in objects:
            d = dict(obj)
            # Convert numpy arrays
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    d[k] = v.tolist()
                elif isinstance(v, set):
                    d[k] = list(v)
            result.append(d)
        return result

    @property
    def session_dir(self) -> str | None:
        return str(self._session_dir) if self._session_dir else None
