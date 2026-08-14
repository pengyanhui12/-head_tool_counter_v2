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
    assert cfg["independent_cooccurrence_max_containment"] == 0.25
    assert cfg["partial_duplicate_min_containment"] == 0.75
    assert cfg["partial_duplicate_max_normalized_distance"] == 0.75
    assert cfg["partial_duplicate_max_absolute_distance_px"] == 80.0
    assert cfg["partial_duplicate_min_mapping_quality"] == 0.50
    assert cfg["partial_duplicate_max_area_ratio"] == 0.60
    assert cfg["partial_duplicate_min_candidate_margin"] == 0.15


def test_offline_associator_assembly_propagates_partial_duplicate_config():
    from apps import offline_scan

    acfg = {
        "min_observations_confirmed": 5,
        "min_keyframes_confirmed": 3,
        "independent_cooccurrence_max_containment": 0.22,
        "partial_duplicate_min_containment": 0.81,
        "partial_duplicate_max_normalized_distance": 0.62,
        "partial_duplicate_max_absolute_distance_px": 71.0,
        "partial_duplicate_min_mapping_quality": 0.58,
        "partial_duplicate_max_area_ratio": 0.47,
        "partial_duplicate_min_candidate_margin": 0.19,
    }

    associator = offline_scan._build_associator(acfg)
    evaluator_config = associator._partial_duplicate_evaluator.config

    assert evaluator_config is associator._partial_duplicate_config
    assert evaluator_config.min_observations_confirmed == 5
    assert evaluator_config.min_keyframes_confirmed == 3
    assert evaluator_config.min_containment == 0.81
    assert evaluator_config.max_normalized_distance == 0.62
    assert evaluator_config.max_absolute_distance_px == 71.0
    assert evaluator_config.min_mapping_quality == 0.58
    assert evaluator_config.max_area_ratio == 0.47
    assert evaluator_config.min_candidate_margin == 0.19
    assert associator.independent_cooccurrence_max_containment == 0.22


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
