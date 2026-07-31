"""对象关联器 — 增强版：空间区域增长 + track 弱关联

核心修改：
- 新增 _try_assign_to_existing_by_region(): polygon overlap 弱关联
- 新增 _try_assign_by_direction_consistency(): 同方向空间的弱关联
- 移除所有"new track → new object"的硬编码，改为 region-based 决策
- 允许多个 track 共享同一个 object（track_ids 集合增长）
"""

# ... (existing code remains unchanged until Phase 2 helper) ...

    def _try_assign_to_existing_objects(
        self,
        gd: GlobalDetection,
        frame_id: int,
        excluded_object_ids: set[str] = None,
    ) -> str | None:
        """尝试用空间约束将 detection 分配给已有的同类对象。

        规则：
        1. 同一 track_id → strong match（已由调用方处理）
        2. Projected polygon 与已有 object 的 bounding region 重叠 > threshold → weak match
        3. Centroid 在 object 的 5-frame 运动方向上 → direction match
        4. 否则返回 None

        Returns:
            object.provisional_id if matched, else None
        """
        excluded = excluded_object_ids or set()

        best_object_id = None
        best_score = 0.0

        for obj in self.map.get_all():
            if obj.provisional_id in excluded:
                continue
            if not self._class_compatible(obj.class_name, gd.class_name):
                continue

            # Rule 2: Polygon overlap
            overlap_score = self._compute_region_overlap(gd, obj)
            if overlap_score > best_score:
                best_score = overlap_score
                best_object_id = obj.provisional_id

            # Rule 3: Direction consistency (if object has ≥3 observations)
            if len(obj.observations) >= 3:
                dir_score = self._compute_direction_consistency(gd, obj)
                if dir_score > best_score:
                    best_score = dir_score
                    best_object_id = obj.provisional_id

        if best_object_id is not None and best_score > 0.15:
            return best_object_id

        return None

    def _compute_region_overlap(
        self, gd: GlobalDetection, obj: GlobalObject
    ) -> float:
        """计算新 detection 与已有 object 的 bounding region 重叠度。

        用 5-quantile 统计来估算 object 的 spatial extent。
        Returns: 0.0–1.0 重叠分数
        """
        if len(obj.observations) < 2:
            return 0.0

        obj_centroids = np.array([
            [obs.polygon_centroid[0], obs.polygon_centroid[1]]
            for obs in obj.observations
        ])
        obj_areas = np.array([obs.polygon_area for obs in obj.observations])

        # Object spatial extent from observations
        cx_min, cy_min = np.percentile(obj_centroids, 5, axis=0)
        cx_max, cy_max = np.percentile(obj_centroids, 95, axis=0)
        region_radius = max(cx_max - cx_min, cy_max - cy_min, 30.0)

        # New detection centroid
        det_cx, det_cy = gd.polygon_centroid

        # Distance to object's region
        dist_to_region_x = max(0.0, abs(det_cx - (cx_min + cx_max) / 2) - (cx_max - cx_min) / 2)
        dist_to_region_y = max(0.0, abs(det_cy - (cy_min + cy_max) / 2) - (cy_max - cy_min) / 2)
        dist_to_region = np.sqrt(dist_to_region_x**2 + dist_to_region_y**2)

        if dist_to_region < region_radius * 0.5:
            return 0.6
        elif dist_to_region < region_radius:
            return 0.3
        else:
            return 0.0

    def _compute_direction_consistency(
        self, gd: GlobalDetection, obj: GlobalObject
    ) -> float:
        """检查新 detection 是否位于 object 的运动方向上。

        用最后 5 帧的质心位移来估算运动方向。
        Returns: 0.0–1.0 一致性分数
        """
        if len(obj.observations) < 3:
            return 0.0

        recent = sorted(obj.observations[-5:], key=lambda o: o.frame_id)
        positions = np.array([
            [obs.polygon_centroid[0], obs.polygon_centroid[1]]
            for obs in recent
        ])

        # Compute velocity from last 2 positions
        if len(positions) >= 2:
            velocity = positions[-1] - positions[-2]
        else:
            velocity = np.array([0.0, 0.0])

        if np.linalg.norm(velocity) < 1.0:
            return 0.0

        # Projected position based on velocity
        projected = positions[-1] + velocity
        det_centroid = np.array([gd.polygon_centroid[0], gd.polygon_centroid[1]])

        # How close is the detection to the projected position?
        dist = np.linalg.norm(det_centroid - projected)
        max_dist = np.linalg.norm(velocity) * 3.0  # Allow ±3x displacement
        if max_dist > 0:
            score = max(0.0, 1.0 - dist / max_dist)
            return score * 0.6  # Weighted lower than region overlap

        return 0.0
