"""全局投影器 — 检测框四角点投影至全局坐标"""
import numpy as np
from shapely.geometry import Polygon

from core.types import RawDetection, GlobalDetection


class GlobalProjector:
    @staticmethod
    def _apply_homography(pt: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
        p = np.array([pt[0], pt[1], 1.0])
        q = H @ p
        if abs(q[2]) < 1e-8:
            return (float("inf"), float("inf"))
        return (float(q[0] / q[2]), float(q[1] / q[2]))

    @staticmethod
    def project(
        detection: RawDetection,
        H_keyframe_to_global: np.ndarray,
        transform_version: int,
    ) -> GlobalDetection:
        # Project four corners -> quadrilateral
        corners = np.array([
            GlobalProjector._apply_homography(c, H_keyframe_to_global)
            for c in detection.corners
        ])

        # Project center separately (NOT a polygon vertex)
        center = GlobalProjector._apply_homography(detection.center, H_keyframe_to_global)

        # Compute polygon centroid and area
        try:
            poly = Polygon(corners.tolist())
            centroid = (poly.centroid.x, poly.centroid.y)
            area = poly.area
        except Exception:
            centroid = tuple(corners.mean(axis=0))
            area = 0.0

        # Edge quality: lower if bbox touches image edge
        edge_quality = 1.0

        return GlobalDetection(
            frame_id=detection.frame_id,
            keyframe_id=detection.keyframe_id,
            track_id=detection.track_id,
            projected_corners=corners,
            projected_center=center,
            polygon_centroid=centroid,
            polygon_area=area,
            class_id=detection.class_id,
            class_name=detection.class_name,
            detection_confidence=detection.confidence,
            sharpness=detection.sharpness,
            mapping_quality=detection.mapping_quality,
            edge_quality=edge_quality,
            size_quality=1.0,
            transform_version=transform_version,
            source=detection.source,
        )

    @staticmethod
    def project_frame_corners(
        image_shape: tuple[int, int],
        H_keyframe_to_global: np.ndarray,
    ) -> Polygon:
        """Project entire frame corners for coverage calculation."""
        h, w = image_shape[:2]
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        projected = [
            GlobalProjector._apply_homography(c, H_keyframe_to_global)
            for c in corners
        ]
        try:
            return Polygon(projected)
        except Exception:
            return Polygon()
