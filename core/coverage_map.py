"""覆盖地图 — 后处理覆盖率检查"""
import numpy as np
from shapely.geometry import Polygon, Point


class CoverageMap:
    def __init__(self, grid_resolution: int = 100):
        self.grid_resolution = grid_resolution
        self._polys: list[Polygon] = []
        self._bounds: tuple[float, float, float, float] | None = None

    def update(self, frame_id: int, projected_fov_polygon: Polygon) -> None:
        if projected_fov_polygon.is_empty or projected_fov_polygon.area <= 0:
            return
        self._polys.append(projected_fov_polygon)

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
