"""全局拼接可视化 — 将关键帧 warped 到全局画布，绘制检测框和对象标注"""
from pathlib import Path

import cv2
import numpy as np

from core.types import GlobalObject
from core.report_generator import get_display_objects


def generate_global_mosaic(
    video_path: str,
    graph,
    objects: list[GlobalObject],
    output_dir: str,
    max_keyframes: int = 20,
    skip_warp: bool = False,
) -> str | None:
    """生成全局拼接图。

    Args:
        video_path: 原始视频路径
        skip_warp: 跳过 warp 帧纹理，只画对象在全局坐标系下的 centroid 和 bbox
        graph: HomographyGraph 实例
        objects: GlobalObject 列表
        output_dir: 输出目录
        max_keyframes: 最多绘制多少个关键帧
    """
    if graph.num_keyframes == 0:
        return None

    objects = get_display_objects(objects)

    nodes = graph.nodes
    if not nodes:
        return None

    if skip_warp:
        h_img, w_img = 720, 1280
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, sample = cap.read()
        if not ok:
            cap.release()
            return None
        h_img, w_img = sample.shape[:2]

    # 计算全局边界
    step = max(1, len(nodes) // max_keyframes)
    corners_local = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=float)
    all_pts = []
    for _node_id, _frame_id, H_to_global in nodes[::step]:
        ones = np.ones((4, 1))
        proj = (H_to_global @ np.hstack([corners_local, ones]).T).T
        denom = proj[:, 2]
        valid = np.abs(denom) > 1e-8
        proj[valid, :2] /= denom[valid, None]
        proj[~valid, :2] = 0
        all_pts.append(proj[:, :2])

    if not all_pts:
        if not skip_warp:
            cap.release()
        return None

    all_pts = np.vstack(all_pts)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    margin = 100
    canvas_w = int(x_max - x_min) + 2 * margin
    canvas_h = int(y_max - y_min) + 2 * margin
    canvas_w = max(canvas_w, 400)
    canvas_h = max(canvas_h, 300)

    offset_x = -x_min + margin
    offset_y = -y_min + margin

    max_canvas = 4000
    scale = 1.0
    if max(canvas_h, canvas_w) > max_canvas:
        scale = max_canvas / max(canvas_h, canvas_w)
        canvas_h = int(canvas_h * scale)
        canvas_w = int(canvas_w * scale)
        offset_x *= scale
        offset_y *= scale

    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    # Warp keyframe textures
    if not skip_warp:
        sampled = nodes[::step]
        for _node_id, frame_id, H_to_global in sampled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            T = np.array([
                [scale, 0, offset_x * scale],
                [0, scale, offset_y * scale],
                [0, 0, 1],
            ])
            M = T @ H_to_global
            warped = cv2.warpPerspective(
                frame, M, (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            alpha = 0.25
            mask = warped.sum(axis=2) > 0
            canvas[mask] = cv2.addWeighted(
                canvas[mask], 1 - alpha, warped[mask], alpha, 0
            ).astype(np.uint8)
        cap.release()

    # Draw grid
    grid_step = 200
    for gx in range(0, canvas_w, grid_step):
        cv2.line(canvas, (gx, 0), (gx, canvas_h), (60, 60, 60), 1)
    for gy in range(0, canvas_h, grid_step):
        cv2.line(canvas, (0, gy), (canvas_w, gy), (60, 60, 60), 1)

    # 收集所有对象的 projected_corners 范围（用于自动缩放）
    all_corners_global = []
    for obj in objects:
        for obs in obj.observations:
            corners = getattr(obs, 'projected_corners', None)
            if corners is not None and len(corners) >= 4:
                all_corners_global.append(np.array(corners))
        cx, cy = obj.centroid_xy
        all_corners_global.append(np.array([[cx, cy]]))

    if all_corners_global:
        all_pts2 = np.vstack(all_corners_global)
        x_min2, y_min2 = all_pts2.min(axis=0)
        x_max2, y_max2 = all_pts2.max(axis=0)

        # 用对象范围重新计算画布，使对象填满画面
        margin2 = 50
        new_w = int(x_max2 - x_min2) + 2 * margin2
        new_h = int(y_max2 - y_min2) + 2 * margin2
        new_w = max(new_w, 400)
        new_h = max(new_h, 300)

        # 计算新的 scale 和 offset
        new_scale = 1.0
        if max(new_h, new_w) > 2000:
            new_scale = 2000.0 / max(new_h, new_w)
            new_h = int(new_h * new_scale)
            new_w = int(new_w * new_scale)

        # 缩放并移位
        new_offset_x = -x_min2 + margin2
        new_offset_y = -y_min2 + margin2

        # 用新参数重建画布
        canvas2 = np.full((new_h, new_w, 3), 40, dtype=np.uint8)

        # Draw grid on zoomed canvas
        for gx in range(0, new_w, grid_step // 2):
            cv2.line(canvas2, (gx, 0), (gx, new_h), (60, 60, 60), 1)
        for gy in range(0, new_h, grid_step // 2):
            cv2.line(canvas2, (0, gy), (new_w, gy), (60, 60, 60), 1)

        # Warp frames if needed
        if not skip_warp:
            cap2 = cv2.VideoCapture(video_path)
            for _node_id, frame_id, H_to_global in nodes[::step]:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap2.read()
                if not ok:
                    continue
                T = np.array([
                    [new_scale, 0, new_offset_x * new_scale],
                    [0, new_scale, new_offset_y * new_scale],
                    [0, 0, 1],
                ])
                M = T @ H_to_global
                warped = cv2.warpPerspective(
                    frame, M, (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_TRANSPARENT,
                )
                alpha = 0.25
                mask = warped.sum(axis=2) > 0
                canvas2[mask] = cv2.addWeighted(
                    canvas2[mask], 1 - alpha, warped[mask], alpha, 0
                ).astype(np.uint8)
            cap2.release()

        # Draw objects on zoomed canvas
        # Draw objects on zoomed canvas
        colors = [
            (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (0, 128, 255), (255, 128, 0), (128, 255, 0), (255, 0, 128),
            (0, 200, 100), (200, 0, 200), (200, 200, 0), (0, 100, 255),
        ]
        for i, obj in enumerate(objects):
            color = colors[i % len(colors)]

            all_corners_list = []
            for obs in obj.observations:
                corners = getattr(obs, 'projected_corners', None)
                if corners is not None and len(corners) >= 4:
                    all_corners_list.append(np.array(corners))

            cx, cy = obj.centroid_xy
            px = int(cx * new_scale + new_offset_x * new_scale)
            py = int(cy * new_scale + new_offset_y * new_scale)

            if all_corners_list:
                for corners_np in all_corners_list[-10:]:
                    px_corners = np.array([
                        [int(c[0] * new_scale + new_offset_x * new_scale),
                         int(c[1] * new_scale + new_offset_y * new_scale)]
                        for c in corners_np
                    ], dtype=np.int32)
                    overlay = canvas2.copy()
                    cv2.polylines(overlay, [px_corners], True, color, 1)
                    cv2.addWeighted(overlay, 0.3, canvas2, 0.7, 0, canvas2)

            cv2.circle(canvas2, (px, py), 5, color, -1)
            pid = obj.persistent_id or obj.provisional_id
            cv2.putText(
                canvas2, f"{pid} {obj.class_name}",
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )

        out = Path(output_dir)
        (out / "global").mkdir(parents=True, exist_ok=True)
        filepath = out / "global" / "global_mosaic.jpg"
        cv2.imwrite(str(filepath), canvas2)

        return str(filepath)
