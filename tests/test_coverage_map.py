"""CoverageMap 单元测试"""
from shapely.geometry import Polygon
from core.coverage_map import CoverageMap


def test_empty_coverage():
    cm = CoverageMap(grid_resolution=50)
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert cm.get_coverage(region) == 0.0


def test_full_coverage():
    cm = CoverageMap(grid_resolution=20)
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    cm.update(0, poly)
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert cm.get_coverage(region) > 0.95


def test_partial_coverage():
    cm = CoverageMap(grid_resolution=50)
    poly = Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])
    cm.update(0, poly)
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    cov = cm.get_coverage(region)
    assert 0.40 < cov < 0.60
