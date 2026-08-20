# Mosaic Single-Warp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在基本不改变 `global_mosaic.jpg` 视觉效果的前提下，删除未参与最终输出的第一遍画布和 Warp，使每个选中关键帧最多执行一次透视变换。

**Architecture:** 保持 `generate_global_mosaic()` 公共接口不变，在 `core/global_mosaic.py` 内提取抽样、边界计算、画布规划、纹理融合和标注绘制等私有函数。最终画布仍由对象投影范围确定，视频只打开一次，选中帧仍按原节点顺序融合。

**Tech Stack:** Python 3.10、OpenCV、NumPy、pytest。

## Global Constraints

- 新增或修改代码必须添加中文注释或中文文档字符串，解释业务意图、算法和边界。
- 不改变对象识别、全局关联、投影坐标及计数逻辑。
- 保持 `generate_global_mosaic(video_path, graph, objects, output_dir, max_keyframes=20, skip_warp=False)` 的签名和返回语义。
- 保持边距 50、最小画布 400×300、最大边长 2000、背景值 40、网格、`alpha=0.25`、对象颜色、最近 10 个轮廓、标签及输出路径。
- 不引入新依赖，不实现局部 ROI Warp，不缓存完整视频帧。
- 未经用户授权不执行 Git commit；每项任务以 `git diff --check` 作为检查点。

---

### Task 1: 用测试锁定单次读取与单次 Warp 契约

**Files:**
- Create: `tests/test_global_mosaic.py`
- Read: `core/global_mosaic.py`

**Interfaces:**
- Consumes: `generate_global_mosaic(video_path, graph, objects, output_dir, max_keyframes=20, skip_warp=False) -> str | None`。
- Produces: 视频打开次数、Warp 次数、画布参数和 `skip_warp` 行为的回归保护。

- [x] **Step 1: 创建确定性测试替身**

在 `tests/test_global_mosaic.py` 定义 `_Graph`、`_Capture`、`_CaptureFactory` 和 `_display_object_with_projected_box()`。Capture 必须实现 `isOpened/set/read/release`，返回固定的 100×160 图像；对象只提供 `observations`、`centroid_xy`、身份和类别属性。

- [x] **Step 2: 写重复 Warp 的失败测试**

```python
def test_mosaic_opens_video_once_and_warps_each_sampled_node_once(
    tmp_path, monkeypatch
):
    graph = _Graph([(0, 0, np.eye(3)), (1, 10, np.eye(3))])
    obj = _display_object_with_projected_box(20, 20, 120, 80)
    captures = _CaptureFactory(np.full((100, 160, 3), 80, np.uint8))
    warp_calls = []
    monkeypatch.setattr(cv2, "VideoCapture", captures)
    monkeypatch.setattr(
        cv2, "warpPerspective",
        lambda frame, matrix, size, **kwargs: (
            warp_calls.append((matrix.copy(), size, kwargs))
            or np.full((size[1], size[0], 3), 80, np.uint8)
        ),
    )
    monkeypatch.setattr(cv2, "imwrite", lambda *_args: True)

    result = generate_global_mosaic("video.mp4", graph, [obj], str(tmp_path))

    assert result == str(tmp_path / "global" / "global_mosaic.jpg")
    assert captures.open_count == 1
    assert len(warp_calls) == 2
```

- [x] **Step 3: 写画布兼容与 skip_warp 测试**

断言对象范围 100×50 时最终 Warp 尺寸为 `(400, 300)`，flags 为 `cv2.INTER_LINEAR`，borderMode 为 `cv2.BORDER_TRANSPARENT`。另一个测试使用 `pytest.fail()` 替换 VideoCapture 和 warpPerspective，并断言 `skip_warp=True` 仍返回输出路径。

- [x] **Step 4: 运行测试确认 RED**

Run: `& 'C:\Users\PC\.conda\envs\head_tool_counter\python.exe' -m pytest tests/test_global_mosaic.py -v`

Expected: 单次打开/单次 Warp 测试失败；当前两节点路径打开视频两次并执行四次 Warp。

- [x] **Step 5: 检查测试差异**

Run: `git diff --check -- tests/test_global_mosaic.py`

Expected: exit code 0。

---

### Task 2: 实现单次画布规划与单次 Warp

**Files:**
- Modify: `core/global_mosaic.py:1-239`
- Test: `tests/test_global_mosaic.py`
- Test: `tests/test_output_identity.py`

**Interfaces:**
- Consumes: `graph.nodes`、`GlobalObject.observations`、`GlobalObject.centroid_xy`。
- Produces: 稳定的公共函数，以及私有 `_sample_nodes()`、`_collect_object_points()`、`_plan_canvas()`、`_warp_sampled_frames()`、`_draw_grid()`、`_draw_objects()`。

- [x] **Step 1: 实现节点抽样和有限点收集**

```python
def _sample_nodes(nodes: list[tuple], max_keyframes: int) -> list[tuple]:
    """沿用原抽样规则，保证关键帧集合和融合顺序不变。"""
    if max_keyframes <= 0:
        raise ValueError("max_keyframes must be positive")
    step = max(1, len(nodes) // max_keyframes)
    return nodes[::step]


def _collect_object_points(objects: list[GlobalObject]) -> np.ndarray | None:
    """只收集有限投影角点和质心，避免异常画布尺寸。"""
    groups = []
    for obj in objects:
        for obs in obj.observations:
            corners = getattr(obs, "projected_corners", None)
            if corners is None or len(corners) < 4:
                continue
            points = np.asarray(corners, dtype=float)
            if points.ndim == 2 and points.shape[1] == 2:
                groups.append(points[np.all(np.isfinite(points), axis=1)])
        centroid = np.asarray(obj.centroid_xy, dtype=float).reshape(1, 2)
        if np.all(np.isfinite(centroid)):
            groups.append(centroid)
    valid = [points for points in groups if len(points)]
    return np.vstack(valid) if valid else None
```

- [x] **Step 2: 实现最终画布规划**

```python
def _plan_canvas(points: np.ndarray) -> tuple[int, int, float, float, float]:
    """按当前最终画布规则返回宽、高、缩放和全局偏移。"""
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
```

- [x] **Step 3: 实现单次视频打开与 Warp**

`_warp_sampled_frames()` 只创建一个 VideoCapture，用 `try/finally` 保证 release；对每个节点执行一次 set/read，读取失败则 continue。变换矩阵保持：

```python
transform = np.array([
    [scale, 0, offset_x * scale],
    [0, scale, offset_y * scale],
    [0, 0, 1],
])
warped = cv2.warpPerspective(
    frame, transform @ h_to_global, (width, height),
    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
)
mask = warped.sum(axis=2) > 0
canvas[mask] = cv2.addWeighted(
    canvas[mask], 0.75, warped[mask], 0.25, 0
).astype(np.uint8)
```

视频无法打开时返回 `False`；调用者在非 skip 模式下据此返回 `None`。

- [x] **Step 4: 提取绘图函数并保持参数**

先以 100 像素间隔画最终网格，再融合纹理，最后绘制对象。保留当前颜色表、最近 10 个 `projected_corners`、轮廓透明度 0.3、半径 5 的质心和字号 0.35 的身份/类别标签。

- [x] **Step 5: 重写 generate_global_mosaic 编排**

流程固定为：检查节点 → 过滤展示对象 → 抽样 → 收集对象点 → 规划最终画布 → 创建背景值 40 的画布 → 网格 → 可选单次 Warp → 对象 → 创建 `global/` → 写入 `global_mosaic.jpg`。删除旧的关键帧角点画布、第一遍 Warp 和第二个 VideoCapture。

- [x] **Step 6: 运行针对性测试确认 GREEN**

Run: `& 'C:\Users\PC\.conda\envs\head_tool_counter\python.exe' -m pytest tests/test_global_mosaic.py tests/test_output_identity.py -v`

Expected: 全部通过，现有 provisional ID 标签测试保持通过。

- [x] **Step 7: 检查实现差异**

Run: `git diff --check -- core/global_mosaic.py tests/test_global_mosaic.py`

Expected: exit code 0。

---

### Task 3: 边界、文档与完整验收

**Files:**
- Modify: `tests/test_global_mosaic.py`
- Modify: `PROJECT_FLOW_REFERENCE.md`
- Verify: `core/global_mosaic.py`

**Interfaces:**
- Consumes: Task 2 的私有函数和稳定公共接口。
- Produces: 异常输入保护、性能/视觉对比和更新后的参考手册。

- [x] **Step 1: 增加边界测试**

增加确定性测试覆盖：无节点返回 None；无有限对象坐标返回 None；视频无法打开返回 None 并 release；两个关键帧中一个读取失败时仍写图且只 Warp 成功读取的一个；`max_keyframes <= 0` 抛出 ValueError。

- [x] **Step 2: 运行针对性与全量测试**

Run:

```powershell
& 'C:\Users\PC\.conda\envs\head_tool_counter\python.exe' -m pytest tests/test_global_mosaic.py tests/test_output_identity.py -v
& 'C:\Users\PC\.conda\envs\head_tool_counter\python.exe' -m pytest
```

Expected: 两条命令均 exit code 0，无失败或收集错误。

- [x] **Step 3: 更新项目参考手册**

在 `PROJECT_FLOW_REFERENCE.md` 的 Mosaic 阶段注明：最终画布先由对象投影范围规划，视频只打开一次，每个抽样关键帧只 Warp 一次。不得写入未经实际测量的固定性能数字。

- [x] **Step 4: 执行同视频性能与视觉对比**

Run: `& 'C:\Users\PC\.conda\envs\head_tool_counter\python.exe' -m apps.offline_scan --output-dir .test-output-mosaic-single-warp --performance`

比较 `outputs12` 与新输出的 `performance.json` 中 `stages.mosaic.total_ms`。用 OpenCV 比较两张 Mosaic 的 shape、最大绝对像素差和平均绝对像素差；比较 report.json 的总数、类别计数、对象 ID、确认状态和审核标记，预期语义一致。

- [x] **Step 5: 最终检查与安全清理**

运行 `git diff --check` 和 `git status --short`。确认临时目录解析后的绝对路径位于仓库内且目录名严格等于 `.test-output-mosaic-single-warp` 后删除；不得删除 `outputs11`、`outputs12` 或其他用户输出。
