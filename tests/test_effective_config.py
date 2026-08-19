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
    assert cfg["end_window_match_candidates"] == 6


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
    assert cfg["feature_cache_size"] == 4


def test_offline_matcher_assembly_propagates_feature_cache_size(monkeypatch):
    from apps import offline_scan

    captured = {}

    class RecordingMatcher:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(offline_scan, "FeatureMatcher", RecordingMatcher)

    offline_scan._build_matcher({"feature_cache_size": 7})

    assert captured["feature_cache_size"] == 7


def test_offline_selector_assembly_propagates_end_window_candidates(
    monkeypatch,
):
    from apps import offline_scan

    captured = {}

    class RecordingSelector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(offline_scan, "KeyframeSelector", RecordingSelector)

    matcher = object()
    offline_scan._build_keyframe_selector(
        {
            "max_keyframe_interval_frames": 30,
            "end_window_frames": 30,
            "end_window_match_candidates": 8,
        },
        matcher,
    )

    assert captured["matcher"] is matcher
    assert captured["end_window_match_candidates"] == 8


def test_config_loader_loads_associator():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.associator
    assert "max_position_distance_px" in cfg
    assert cfg["online_gate_ratio"] == 0.5
    assert "per_class_gate_ratios" in cfg
    assert "per_class_position_gates" in cfg
    assert cfg["class_compatibility"]
    assert cfg["independent_co_occurrence_max_containment"] == 0.25
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
        "independent_co_occurrence_max_containment": 0.22,
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
    assert associator.independent_co_occurrence_max_containment == 0.22


def test_config_loader_loads_fusion():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg = loader.detector["fusion"]
    assert cfg["iou_threshold"] == 0.65
    assert cfg["center_merge_distance_px"] == 40.0
    assert cfg["center_merge_min_ios"] == 0.30


def test_offline_detector_assembly_propagates_all_detector_config(monkeypatch):
    """正式流水线装配必须传递模型、设备和 L1/L2/L3 参数。"""
    from apps import offline_scan

    captured = {}

    class RecordingDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(offline_scan, "Detector", RecordingDetector)
    detector_cfg = {
        "model": {"path": "weights/custom.pt", "device": "cpu"},
        "levels": {
            "L1": {"imgsz": 1024, "conf": 0.21, "iou": 0.61},
            "L2": {"imgsz": 512, "conf": 0.12, "iou": 0.62},
            "L3": {"imgsz": 896, "conf": 0.13, "iou": 0.63},
        },
    }

    project_root = Path("D:/project-root")
    detector = offline_scan._build_detector(detector_cfg, project_root)

    assert isinstance(detector, RecordingDetector)
    assert captured == {
        "model_path": str(project_root / "weights" / "custom.pt"),
        "device": "cpu",
        "l1_imgsz": 1024,
        "l1_conf": 0.21,
        "l1_iou": 0.61,
        "l2_imgsz": 512,
        "l2_conf": 0.12,
        "l2_iou": 0.62,
        "l3_imgsz": 896,
        "l3_conf": 0.13,
        "l3_iou": 0.63,
    }


def test_offline_detector_assembly_preserves_absolute_model_path(monkeypatch):
    """绝对模型路径不能再次拼接项目根目录。"""
    from apps import offline_scan

    captured = {}
    absolute_model = Path("D:/models/custom.pt")

    class RecordingDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(offline_scan, "Detector", RecordingDetector)
    offline_scan._build_detector(
        {"model": {"path": str(absolute_model), "device": "cuda:1"}},
        Path("ignored-root"),
    )

    assert captured["model_path"] == str(absolute_model)
    assert captured["device"] == "cuda:1"


def test_offline_coverage_assembly_propagates_all_coverage_config():
    """覆盖地图必须使用 YAML 中的分辨率、最小面积和目标覆盖率。"""
    from apps import offline_scan

    coverage = offline_scan._build_coverage_map({
        "grid_resolution": 37,
        "minimum_valid_polygon_area": 123.0,
        "target_coverage_ratio": 0.87,
    })

    assert coverage.grid_resolution == 37
    assert coverage.minimum_valid_polygon_area == 123.0
    assert coverage.target_coverage_ratio == 0.87


def test_config_loader_caches():
    loader = ConfigLoader(Path(__file__).parent.parent / "configs")
    cfg1 = loader.pipeline
    cfg2 = loader.pipeline
    assert cfg1 is cfg2  # Same cached dict
