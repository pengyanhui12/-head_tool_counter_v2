"""有效配置加载测试 — ConfigLoader 正确合并默认值"""
import numpy as np
import pytest
from pathlib import Path
import yaml

from core.config_loader import ConfigLoader


def test_config_loader_loads_pipeline():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.pipeline
    assert "l2_interval_frames" in cfg
    assert "max_keyframe_interval_frames" in cfg
    assert "min_keyframe_interval_frames" in cfg


def test_config_loader_loads_tracker():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.tracker
    assert "min_iou" in cfg
    assert "max_center_distance_ratio" in cfg
    assert "quality_drop_trigger_ratio" in cfg


def test_config_loader_loads_matcher():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.matcher
    assert "min_good_matches" in cfg
    assert "min_inliers" in cfg


def test_config_loader_loads_associator():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.associator
    assert "max_position_distance_px" in cfg
    # class_compatibility 在顶层，不在 association 里
    assert loader.load("associator").get("class_compatibility") is not None


def test_config_loader_caches():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg1 = loader.pipeline
    cfg2 = loader.pipeline
    assert cfg1 is cfg2  # Same cached dict
