"""配置加载器——从 configs/ 目录读取 YAML 配置"""
from __future__ import annotations

from pathlib import Path

import yaml


class ConfigLoader:
    """加载并缓存所有配置文件。"""

    def __init__(self, config_dir: str | Path = "configs"):
        self._config_dir = Path(config_dir)
        self._cache: dict[str, dict] = {}

    def load(self, name: str) -> dict:
        """加载 configs/<name>.yaml，自动缓存。"""
        if name not in self._cache:
            path = self._config_dir / f"{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, encoding="utf-8") as f:
                self._cache[name] = yaml.safe_load(f)
        return self._cache[name]

    @property
    def pipeline(self) -> dict:
        return self.load("pipeline")["pipeline"]

    @property
    def detector(self) -> dict:
        return self.load("detector")

    @property
    def tracker(self) -> dict:
        return self.load("tracker")["tracker"]

    @property
    def matcher(self) -> dict:
        return self.load("matcher")["matcher"]

    @property
    def associator(self) -> dict:
        return self.load("associator")["association"]

    @property
    def coverage(self) -> dict:
        return self.load("coverage")["coverage"]

    @property
    def camera(self) -> dict:
        return self.load("camera")["camera"]
