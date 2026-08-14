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
    assert cfg["quality_evaluation_scale"] == 0.5
    assert cfg["detection_sharpness_threshold"] == 63.0
    assert cfg["sharpness_threshold"] == 89.0
    assert cfg["enable_performance_stats"] is False


def test_config_loader_loads_tracker():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.tracker
    assert "min_iou" in cfg
    assert "max_center_distance_ratio" in cfg
    assert "quality_drop_trigger_ratio" in cfg
    assert cfg["new_detection_confirmation_runs"] == 3


def test_config_loader_loads_matcher():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.matcher
    assert "min_good_matches" in cfg
    assert "min_inliers" in cfg


def test_config_loader_loads_associator():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.associator
    assert "max_position_distance_px" in cfg
    assert cfg["online_gate_ratio"] == 0.5
    assert "per_class_gate_ratios" in cfg
    assert "per_class_position_gates" in cfg
    assert cfg["class_compatibility"]


def test_config_loader_loads_fusion():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.detector["fusion"]
    assert cfg["iou_threshold"] == 0.65
    assert cfg["center_merge_distance_px"] == 40.0
    assert cfg["center_merge_min_ios"] == 0.30


def test_config_loader_caches():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg1 = loader.pipeline
    cfg2 = loader.pipeline
    assert cfg1 is cfg2  # Same cached dict
