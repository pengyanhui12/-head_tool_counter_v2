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


def test_update_ignores_polygon_below_configured_minimum_area():
    """面积不足的投影视野不能污染覆盖率。"""
    cm = CoverageMap(grid_resolution=20, minimum_valid_polygon_area=200.0)
    small = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    cm.update(0, small)

    assert cm.get_coverage(region) == 0.0


def test_update_keeps_rejecting_zero_area_polygon_with_default_threshold():
    """默认最小面积为零时仍必须拒绝退化多边形。"""
    cm = CoverageMap(grid_resolution=20)
    degenerate = Polygon([(0, 0), (10, 0), (20, 0)])
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    cm.update(0, degenerate)

    assert cm._polys == []
    assert cm.get_coverage(region) == 0.0


def test_target_reached_uses_configured_ratio():
    """目标覆盖判断必须使用配置阈值。"""
    cm = CoverageMap(grid_resolution=20, target_coverage_ratio=0.80)
    covered = Polygon([(0, 0), (90, 0), (90, 100), (0, 100)])
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    cm.update(0, covered)

    assert cm.is_target_reached(region) is True
