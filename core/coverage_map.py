"""覆盖地图 — 后处理覆盖率检查"""
import numpy as np
from shapely.geometry import Polygon, Point


class CoverageMap:
    def __init__(
        self,
        grid_resolution: int = 100,
        minimum_valid_polygon_area: float = 0.0,
        target_coverage_ratio: float = 0.95,
    ):
        if grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive")
        if minimum_valid_polygon_area < 0:
            raise ValueError("minimum_valid_polygon_area must be non-negative")
        if not 0.0 <= target_coverage_ratio <= 1.0:
            raise ValueError("target_coverage_ratio must be in [0, 1]")
        self.grid_resolution = grid_resolution
        self.minimum_valid_polygon_area = minimum_valid_polygon_area
        self.target_coverage_ratio = target_coverage_ratio
        self._polys: list[Polygon] = []
        self._bounds: tuple[float, float, float, float] | None = None

    def update(self, frame_id: int, projected_fov_polygon: Polygon) -> None:
        # 过滤无效或面积过小的投影视野，避免噪声污染覆盖率。
        if (
            projected_fov_polygon.is_empty
            or projected_fov_polygon.area <= 0
            or projected_fov_polygon.area < self.minimum_valid_polygon_area
        ):
            return
        self._polys.append(projected_fov_polygon)

    def is_target_reached(self, region_polygon: Polygon) -> bool:
        """判断指定区域是否已经达到配置的目标覆盖率。"""
        return self.get_coverage(region_polygon) >= self.target_coverage_ratio

    def get_coverage(self, region_polygon: Polygon) -> float:
        if not self._polys:
            return 0.0

        minx, miny, maxx, maxy = region_polygon.bounds
        self._bounds = (minx, miny, maxx, maxy)
        cell_w = (maxx - minx) / self.grid_resolution
        cell_h = (maxy - miny) / self.grid_resolution

        grid = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.int32)

        for poly in self._polys:
            px0, py0, px1, py1 = poly.bounds
            gx0 = max(0, int((px0 - minx) / cell_w))
            gy0 = max(0, int((py0 - miny) / cell_h))
            gx1 = min(self.grid_resolution - 1, int((px1 - minx) / cell_w))
            gy1 = min(self.grid_resolution - 1, int((py1 - miny) / cell_h))

            for gy in range(gy0, gy1 + 1):
                for gx in range(gx0, gx1 + 1):
                    cx = minx + (gx + 0.5) * cell_w
                    cy = miny + (gy + 0.5) * cell_h
                    if poly.contains(Point(cx, cy)):
                        grid[gy, gx] = 1

        total = self.grid_resolution * self.grid_resolution
        covered = int(grid.sum())
        return covered / total
