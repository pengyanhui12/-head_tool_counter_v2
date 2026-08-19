"""全局拼接可视化：在统一坐标系中融合关键帧并绘制对象标注。"""

from pathlib import Path

import cv2
import numpy as np

from core.report_generator import get_display_objects
from core.types import GlobalObject


_OBJECT_COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (128, 255, 0),
    (255, 0, 128),
    (0, 200, 100),
    (200, 0, 200),
    (200, 200, 0),
    (0, 100, 255),
]


def _sample_nodes(nodes: list[tuple], max_keyframes: int) -> list[tuple]:
    """沿用原抽样规则，保证关键帧集合和融合顺序不变。"""
    if isinstance(max_keyframes, bool) or not isinstance(max_keyframes, int):
        raise ValueError("max_keyframes must be a positive integer")
    if max_keyframes <= 0:
        raise ValueError("max_keyframes must be a positive integer")
    step = max(1, len(nodes) // max_keyframes)
    return nodes[::step]


def _collect_object_points(objects: list[GlobalObject]) -> np.ndarray | None:
    """收集有限投影角点和质心，防止异常坐标生成无效画布。"""
    groups: list[np.ndarray] = []
    for obj in objects:
        for observation in obj.observations:
            corners = getattr(observation, "projected_corners", None)
            if corners is None or len(corners) < 4:
                continue
            points = np.asarray(corners, dtype=float)
            if points.ndim == 2 and points.shape[1] == 2:
                finite_points = points[np.all(np.isfinite(points), axis=1)]
                if len(finite_points) > 0:
                    groups.append(finite_points)

        centroid = np.asarray(obj.centroid_xy, dtype=float)
        if centroid.shape == (2,) and np.all(np.isfinite(centroid)):
            groups.append(centroid.reshape(1, 2))

    return np.vstack(groups) if groups else None


def _plan_canvas(
    points: np.ndarray,
) -> tuple[int, int, float, float, float]:
    """按原最终画布规则返回宽、高、缩放与全局坐标偏移。"""
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    margin = 50
    width = max(int(x_max - x_min) + 2 * margin, 400)
    height = max(int(y_max - y_min) + 2 * margin, 300)

    scale = 1.0
    if max(height, width) > 2000:
        scale = 2000.0 / max(height, width)
        height = int(height * scale)
        width = int(width * scale)

    return width, height, scale, -x_min + margin, -y_min + margin


def _draw_grid(canvas: np.ndarray) -> None:
    """沿用原最终画布的100像素网格。"""
    height, width = canvas.shape[:2]
    for x in range(0, width, 100):
        cv2.line(canvas, (x, 0), (x, height), (60, 60, 60), 1)
    for y in range(0, height, 100):
        cv2.line(canvas, (0, y), (width, y), (60, 60, 60), 1)


def _warp_sampled_frames(
    video_path: str,
    sampled_nodes: list[tuple],
    canvas: np.ndarray,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> bool:
    """只打开一次视频，每个抽样节点最多读取并Warp一次。"""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        return False

    try:
        height, width = canvas.shape[:2]
        canvas_transform = np.array(
            [
                [scale, 0, offset_x * scale],
                [0, scale, offset_y * scale],
                [0, 0, 1],
            ]
        )
        for _node_id, frame_id, transform_to_global in sampled_nodes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                # 单个关键帧损坏不应阻止其余帧和对象标注输出。
                continue
            # 透明边界模式不会主动初始化目标区；显式清零可避免随机内存进入融合掩码。
            warp_buffer = np.zeros_like(canvas)
            warped = cv2.warpPerspective(
                frame,
                canvas_transform @ transform_to_global,
                (width, height),
                dst=warp_buffer,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            mask = warped.sum(axis=2) > 0
            canvas[mask] = cv2.addWeighted(
                canvas[mask], 0.75, warped[mask], 0.25, 0
            ).astype(np.uint8)
        return True
    finally:
        capture.release()


def _draw_objects(
    canvas: np.ndarray,
    objects: list[GlobalObject],
    scale: float,
    offset_x: float,
    offset_y: float,
) -> None:
    """绘制最近观测轮廓、对象质心及稳定身份标签。"""
    for index, obj in enumerate(objects):
        color = _OBJECT_COLORS[index % len(_OBJECT_COLORS)]
        projected_observations = []
        for observation in obj.observations:
            corners = getattr(observation, "projected_corners", None)
            if corners is None or len(corners) < 4:
                continue
            points = np.asarray(corners, dtype=float)
            if (
                points.ndim == 2
                and points.shape[1] == 2
                and np.all(np.isfinite(points))
            ):
                projected_observations.append(points)

        for corners in projected_observations[-10:]:
            pixel_corners = np.array(
                [
                    [
                        int(point[0] * scale + offset_x * scale),
                        int(point[1] * scale + offset_y * scale),
                    ]
                    for point in corners
                ],
                dtype=np.int32,
            )
            overlay = canvas.copy()
            cv2.polylines(overlay, [pixel_corners], True, color, 1)
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        centroid_x, centroid_y = obj.centroid_xy
        pixel_x = int(centroid_x * scale + offset_x * scale)
        pixel_y = int(centroid_y * scale + offset_y * scale)
        cv2.circle(canvas, (pixel_x, pixel_y), 5, color, -1)
        identity = obj.persistent_id or obj.provisional_id
        cv2.putText(
            canvas,
            f"{identity} {obj.class_name}",
            (pixel_x + 6, pixel_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )


def generate_global_mosaic(
    video_path: str,
    graph,
    objects: list[GlobalObject],
    output_dir: str,
    max_keyframes: int = 20,
    skip_warp: bool = False,
) -> str | None:
    """生成全局拼接图；纹理只经过一次最终画布Warp。"""
    if graph.num_keyframes == 0:
        return None

    nodes = graph.nodes
    if not nodes:
        return None

    display_objects = get_display_objects(objects)
    object_points = _collect_object_points(display_objects)
    if object_points is None:
        return None

    sampled_nodes = _sample_nodes(nodes, max_keyframes)
    width, height, scale, offset_x, offset_y = _plan_canvas(object_points)
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)

    # 当前视觉效果要求网格先于纹理绘制，纹理会自然覆盖部分网格。
    _draw_grid(canvas)
    if not skip_warp and not _warp_sampled_frames(
        video_path,
        sampled_nodes,
        canvas,
        scale,
        offset_x,
        offset_y,
    ):
        return None
    _draw_objects(canvas, display_objects, scale, offset_x, offset_y)

    output_path = Path(output_dir) / "global" / "global_mosaic.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return str(output_path)
