"""特征匹配器——SIFT + RANSAC + 双图空间分布 + 退化检查"""
from collections import OrderedDict

import cv2
import numpy as np

from core.types import MatchResult


class FeatureMatcher:
    def __init__(
        self,
        mode: str = "sift",
        ratio_test: float = 0.75,
        ransac_threshold_px: float = 3.0,
        min_good_matches: int = 20,
        min_inliers: int = 30,
        min_inlier_ratio: float = 0.30,
        max_reprojection_error: float = 3.0,
        min_occupied_quadrants: int = 3,
        min_inlier_bbox_area_ratio: float = 0.15,
        roi_center_ratio: float = 0.70,
        max_projected_area_ratio: float = 10.0,
        min_projected_area_ratio: float = 0.10,
        max_condition_number: float = 5e5,
        feature_cache_size: int = 4,
    ):
        if (
            not isinstance(feature_cache_size, int)
            or isinstance(feature_cache_size, bool)
            or feature_cache_size < 0
        ):
            raise ValueError("feature_cache_size must be a non-negative integer")
        self.mode = mode
        self.ratio_test = ratio_test
        self.ransac_threshold = ransac_threshold_px
        self.min_good_matches = min_good_matches
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reprojection_error = max_reprojection_error
        self.min_occupied_quadrants = min_occupied_quadrants
        self.min_inlier_bbox_area_ratio = min_inlier_bbox_area_ratio
        self.roi_center_ratio = roi_center_ratio
        self.max_projected_area_ratio = max_projected_area_ratio
        self.min_projected_area_ratio = min_projected_area_ratio
        self.max_condition_number = max_condition_number
        self.feature_cache_size = feature_cache_size
        self._detector = cv2.SIFT_create()
        # 缓存图像强引用以防对象ID复用；小容量限制额外图像内存占用。
        self._feature_cache: OrderedDict[int, tuple[np.ndarray, tuple]] = (
            OrderedDict()
        )

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _roi_mask(self, h: int, w: int) -> np.ndarray:
        r = self.roi_center_ratio
        y0 = int(h * (1 - r) / 2)
        y1 = int(h * (1 + r) / 2)
        x0 = int(w * (1 - r) / 2)
        x1 = int(w * (1 + r) / 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return mask

    def extract_features(self, image: np.ndarray):
        cache_key = id(image)
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            cached_image, cached_features = cached
            if cached_image is image:
                self._feature_cache.move_to_end(cache_key)
                return cached_features
            # 理论上强引用会阻止ID复用；仍保留防御检查以保护缓存契约。
            self._feature_cache.pop(cache_key, None)

        gray = self._to_gray(image)
        mask = self._roi_mask(*gray.shape)
        features = self._detector.detectAndCompute(gray, mask)
        if self.feature_cache_size > 0:
            self._feature_cache[cache_key] = (image, features)
            self._feature_cache.move_to_end(cache_key)
            while len(self._feature_cache) > self.feature_cache_size:
                self._feature_cache.popitem(last=False)
        return features

    def clear_feature_cache(self) -> None:
        """图像可能被原地修改时，由调用方显式清除陈旧特征。"""
        self._feature_cache.clear()

    def match(self, source_image: np.ndarray, target_image: np.ndarray) -> MatchResult:
        kp_src, desc_src = self.extract_features(source_image)
        kp_dst, desc_dst = self.extract_features(target_image)

        h_src, w_src = source_image.shape[:2]
        h_dst, w_dst = target_image.shape[:2]

        def fail(reason: str) -> MatchResult:
            return MatchResult(
                H_source_to_target=None,
                num_keypoints_src=len(kp_src) if kp_src else 0,
                num_keypoints_dst=len(kp_dst) if kp_dst else 0,
                num_good_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                reprojection_error=float("inf"),
                occupied_quadrants_src=0,
                occupied_quadrants_dst=0,
                inlier_bbox_area_ratio_src=0.0,
                inlier_bbox_area_ratio_dst=0.0,
                valid=False,
                failure_reason=reason,
            )

        if not kp_src or not kp_dst or len(kp_src) < 10 or len(kp_dst) < 10:
            return fail(f"insufficient keypoints: src={len(kp_src) if kp_src else 0}, dst={len(kp_dst) if kp_dst else 0}")

        matches = cv2.BFMatcher().knnMatch(desc_src, desc_dst, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good.append(m)

        if len(good) < self.min_good_matches:
            return fail(f"insufficient good matches: {len(good)} < {self.min_good_matches}")

        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)

        if H is None:
            return fail("findHomography returned None")

        # Normalize H[2,2]
        H = H / H[2, 2]

        inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = inliers / len(good)

        # Reprojection error (inliers only)
        ones = np.ones((src_pts.shape[0], 1))
        proj = (H @ np.hstack([src_pts, ones]).T).T
        proj = proj[:, :2] / proj[:, 2:3]
        if mask is not None and inliers > 0:
            reproj_err = float(np.linalg.norm(proj - dst_pts, axis=1)[mask.ravel() == 1].mean())
        else:
            reproj_err = float("inf")

        # Spatial distribution based on RANSAC inliers only
        quad_src, bbox_src = self._inlier_spatial(kp_src, good, mask, w_src, h_src)
        quad_dst, bbox_dst = self._inlier_spatial(kp_dst, good, mask, w_dst, h_dst, use_train=True)

        # Degeneracy check
        degen = self._check_degeneracy(H, src_pts, dst_pts, mask, w_src, h_src)
        if degen:
            return MatchResult(
                H_source_to_target=H,
                num_keypoints_src=len(kp_src),
                num_keypoints_dst=len(kp_dst),
                num_good_matches=len(good),
                num_inliers=inliers,
                inlier_ratio=inlier_ratio,
                reprojection_error=reproj_err,
                occupied_quadrants_src=quad_src,
                occupied_quadrants_dst=quad_dst,
                inlier_bbox_area_ratio_src=bbox_src,
                inlier_bbox_area_ratio_dst=bbox_dst,
                valid=False,
                failure_reason=degen,
            )

        quality_valid = (
            len(good) >= self.min_good_matches
            and inliers >= self.min_inliers
            and inlier_ratio >= self.min_inlier_ratio
            and reproj_err <= self.max_reprojection_error
            and quad_src >= self.min_occupied_quadrants
            and quad_dst >= self.min_occupied_quadrants
            and bbox_src >= self.min_inlier_bbox_area_ratio
            and bbox_dst >= self.min_inlier_bbox_area_ratio
        )

        return MatchResult(
            H_source_to_target=H,
            num_keypoints_src=len(kp_src),
            num_keypoints_dst=len(kp_dst),
            num_good_matches=len(good),
            num_inliers=inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error=reproj_err,
            occupied_quadrants_src=quad_src,
            occupied_quadrants_dst=quad_dst,
            inlier_bbox_area_ratio_src=bbox_src,
            inlier_bbox_area_ratio_dst=bbox_dst,
            valid=quality_valid,
            failure_reason=None if quality_valid else "quality thresholds not met",
        )

    def _inlier_spatial(self, kpts, matches, mask, w, h, use_train=False):
        """Compute occupied quadrants and inlier bbox area ratio from RANSAC inliers."""
        xs, ys = [], []
        for i, (m, ok) in enumerate(zip(matches, (mask.ravel() if mask is not None else []))):
            if not ok:
                continue
            idx = m.trainIdx if use_train else m.queryIdx
            x, y = kpts[idx].pt
            xs.append(x)
            ys.append(y)

        if not xs:
            return 0, 0.0

        # quadrants
        mx, my = w / 2, h / 2
        quads = set()
        for x, y in zip(xs, ys):
            qx = 0 if x < mx else 1
            qy = 0 if y < my else 2
            quads.add(qy + qx)

        # bbox area ratio
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_area = (max_x - min_x) * (max_y - min_y)
        bbox_ratio = bbox_area / (w * h) if (w * h) > 0 else 0.0

        return len(quads), bbox_ratio

    def _check_degeneracy(self, H, src_pts, dst_pts, mask, w, h) -> str | None:
        if np.any(~np.isfinite(H)):
            return "H contains NaN or Inf"

        det = np.linalg.det(H)
        if abs(det) < 1e-6:
            return f"det(H) too small: {det:.2e}"

        cond = np.linalg.cond(H)
        if cond > self.max_condition_number:
            return f"condition number too large: {cond:.1f}"

        if abs(H[2, 0]) > 0.1 or abs(H[2, 1]) > 0.1:
            return f"extreme perspective: H31={H[2,0]:.3f}, H32={H[2,1]:.3f}"

        # Project image corners to check area change
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        ones = np.ones((4, 1))
        proj = (H @ np.hstack([corners, ones]).T).T
        denom = proj[:, 2]
        if np.any(np.abs(denom) < 1e-8):
            return "homogeneous denominator near zero"
        proj = proj[:, :2] / denom[:, None]

        # Check for extreme area change or self-intersection
        try:
            from shapely.geometry import Polygon
            poly = Polygon(proj.tolist())
            if not poly.is_valid or poly.area <= 0:
                return "projected polygon invalid or zero area"
            area_ratio = poly.area / (w * h)
            if area_ratio < self.min_projected_area_ratio or area_ratio > self.max_projected_area_ratio:
                return f"extreme area change: {area_ratio:.4f}x"
        except Exception:
            return "projected polygon computation failed"

        return None
