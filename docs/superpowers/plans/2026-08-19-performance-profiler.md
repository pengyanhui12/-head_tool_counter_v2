# Performance Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将离线性能统计拆到独立模块，修复计时边界和指标歧义，并把机器可读与可读报告保存到输出目录。

**Architecture:** 新建 `core/performance_profiler.py`，由 `PerformanceProfiler` 统一管理阶段定义、可选计时、匹配器包装、快照计算、文本格式化和文件保存。`apps/offline_scan.py` 只在业务边界调用 profiler，并补齐 Recovery 与尾窗成功路径的阶段计时。

**Tech Stack:** Python 3.10+、标准库 `time/json/contextlib/pathlib`、pytest。

## Global Constraints

- 不改变检测、跟踪、关键帧选择、关联、计数和现有报告内容。
- 新增或修改的代码必须添加解释业务意图、边界或算法取舍的中文注释。
- 关闭性能统计时不生成 `performance.json` 或 `performance.txt`。
- 不优化 SIFT 和 Global Mosaic；二者属于后续独立实验。
- 保留用户在 `apps/offline_scan.py` 中对默认视频和 `outputs11` 的本地修改。
- 不执行 Git 提交或推送。

---

### Task 1: 独立性能统计核心

**Files:**
- Create: `core/performance_profiler.py`
- Create: `tests/test_performance_profiler.py`

**Interfaces:**
- Produces: `PerformanceProfiler(enabled: bool, started_at: float | None = None)`。
- Produces: `measure(stage: str)`、`record(stage: str, elapsed_ms: float)`、`measure_success(stage: str, operation: Callable[[], T]) -> T`。
- Produces: `stage_total(stage: str) -> float` 和 `record_exclusive(stage: str, started_at: float, excluded_stage: str, excluded_before_ms: float, ended_at: float | None = None)`。
- Produces: `wrap_matcher(matcher)`、`snapshot(total_frames: int, ended_at: float | None = None) -> dict`、`format_report(snapshot: dict) -> str`、`save(output_dir: str | Path, total_frames: int) -> dict`。

- [x] **Step 1: 编写失败测试**

在 `tests/test_performance_profiler.py` 中覆盖：累计与平均值、未知阶段报错、关闭状态不累计、失败操作不增加成功调用次数、排除嵌套阶段、负未统计差额产生 `timing_overlap_detected`、JSON/TXT 保存和匹配器透明代理。

```python
def test_failed_operation_does_not_count_as_success():
    profiler = PerformanceProfiler(enabled=True, started_at=10.0)

    with pytest.raises(StopIteration):
        profiler.measure_success("video_decode", lambda: next(iter(())))

    snapshot = profiler.snapshot(total_frames=0, ended_at=10.1)
    assert snapshot["stages"]["video_decode"]["calls"] == 0


def test_overlap_is_reported_instead_of_clamped():
    profiler = PerformanceProfiler(enabled=True, started_at=10.0)
    profiler.record("quality", 120.0)

    snapshot = profiler.snapshot(total_frames=1, ended_at=10.1)

    assert snapshot["unaccounted_ms"] == pytest.approx(-20.0)
    assert snapshot["timing_overlap_detected"] is True
    assert snapshot["accounted_coverage"] == pytest.approx(1.2)
```

- [x] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_performance_profiler.py -q`

Expected: FAIL，原因是 `core.performance_profiler` 尚不存在。

- [x] **Step 3: 实现最小核心模块**

模块使用固定阶段键，展示标签与现有表格一致；快照 schema 为：

```python
{
    "schema_version": "1.0",
    "total_frames": 406,
    "pipeline_wall_ms": 24859.0,
    "pipeline_fps": 16.3,
    "accounted_ms": 24687.7,
    "unaccounted_ms": 171.3,
    "accounted_coverage": 0.993,
    "timing_overlap_detected": False,
    "stages": {
        "video_decode": {
            "label": "Video decode",
            "total_ms": 632.0,
            "calls": 406,
            "average_ms": 1.557,
            "wall_percent": 2.54,
        }
    },
}
```

`measure_success()` 只在 operation 正常返回后记录耗时和一次调用；异常原样抛出。`save()` 创建 `<output_dir>/reports`，先生成一次快照，再写 `performance.json` 和 `performance.txt`，文件写盘时间不回写快照。

- [x] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/test_performance_profiler.py -q`

Expected: 全部 PASS。

---

### Task 2: 接入 offline_scan 并修复基础口径

**Files:**
- Modify: `apps/offline_scan.py`
- Modify: `tests/test_offline_performance.py`
- Modify: `tests/test_debug_performance.py`

**Interfaces:**
- Consumes: Task 1 的 `PerformanceProfiler`。
- Produces: `run_pipeline()` 在启用时生成两个性能文件，控制台使用 `Pipeline wall time`。

- [x] **Step 1: 迁移并扩展失败测试**

将 `resolve_performance_enabled` 的导入迁移到新模块；删除对 `apps.offline_scan.print_performance_stats` 的依赖，改为验证 `format_report()`。增加成功/EOF 调用计数测试，要求 406 帧对应 406 次 `video_decode`。

```python
def test_report_uses_unambiguous_pipeline_wall_label():
    profiler = PerformanceProfiler(enabled=True, started_at=1.0)
    text = profiler.format_report(
        profiler.snapshot(total_frames=2, ended_at=1.02)
    )
    assert "Pipeline wall time" in text
    assert "End-to-end" not in text
```

- [x] **Step 2: 验证迁移测试失败**

Run: `python -m pytest tests/test_offline_performance.py tests/test_debug_performance.py -q`

Expected: 至少一个 FAIL，表明旧实现仍从 `offline_scan.py` 暴露或使用旧标签。

- [x] **Step 3: 最小接入新模块**

在 `run_pipeline()` 第一条业务语句创建入口时间，配置加载后创建 profiler；把 initialization 的起点设为入口时间。用 `measure_success("video_decode", ...)` 替代对 `next()` 的普通上下文计时。其余 `with timer(...)` 改为 `with profiler.measure(...)`，匹配器由 `profiler.wrap_matcher()` 包装。删除 `PERFORMANCE_LABELS`、`print_performance_stats()`、局部 `timer()` 和 `DebugStats/PerfTimer/TimedMatcher` 导入。

结束时执行：

```python
snapshot = profiler.save(output_dir=out, total_frames=fc)
print(profiler.format_report(snapshot))
```

- [x] **Step 4: 验证基础接入**

Run: `python -m pytest tests/test_performance_profiler.py tests/test_offline_performance.py tests/test_debug_performance.py -q`

Expected: 全部 PASS。

---

### Task 3: 补齐 Recovery 与尾窗计时

**Files:**
- Modify: `apps/offline_scan.py`
- Modify: `tests/test_offline_performance.py`

**Interfaces:**
- Consumes: `PerformanceProfiler.measure()`。
- Produces: Recovery 和尾窗成功路径与常规关键帧路径使用相同阶段键。

- [x] **Step 1: 编写独占计时回归测试**

在 `tests/test_performance_profiler.py` 验证 `record_exclusive()` 会从外层墙钟耗时中扣除本次新增的 `feature_match` 时间，并只给外层阶段增加一次调用：

```python
def test_record_exclusive_subtracts_nested_stage_time():
    profiler = PerformanceProfiler(enabled=True, started_at=1.0)
    before = profiler.stage_total("feature_match")
    profiler.record("feature_match", 30.0)

    profiler.record_exclusive(
        "recovery",
        started_at=2.0,
        excluded_stage="feature_match",
        excluded_before_ms=before,
        ended_at=2.05,
    )

    snapshot = profiler.snapshot(total_frames=1, ended_at=2.1)
    assert snapshot["stages"]["recovery"]["total_ms"] == pytest.approx(20.0)
    assert snapshot["stages"]["recovery"]["calls"] == 1
```

- [x] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_offline_performance.py -q`

Expected: FAIL，原因是 `record_exclusive()` 尚未实现或尚未正确扣除嵌套时间。

- [x] **Step 3: 为遗漏操作添加计时边界**

Recovery 成功路径分别包裹图更新、L1、融合、Tracker update、投影和关联；尾窗接受路径分别包裹图更新、L1、融合、投影和关联。Recovery 自身耗时继续扣除嵌套 feature matching；尾窗选择与 evaluate 自身耗时同样扣除 feature matching。

- [x] **Step 4: 验证相关测试**

Run: `python -m pytest tests/test_performance_profiler.py tests/test_offline_performance.py tests/test_debug_performance.py tests/test_end_window_keyframes.py tests/test_recovery_parent_graph.py -q`

Expected: 全部 PASS。

---

### Task 4: 文档与完整验证

**Files:**
- Modify: `PROJECT_FLOW_REFERENCE.md`
- Modify: `docs/superpowers/plans/2026-08-19-performance-profiler.md`

**Interfaces:**
- Produces: 当前性能统计开关、输出路径和指标口径的中文参考说明。

- [x] **Step 1: 更新参考手册**

记录 `--performance`、YAML 开关、两个性能文件、Pipeline wall time 范围、核心流水线与产物生成的区别，以及论文实验应固定视频/配置/设备并重复运行。

- [x] **Step 2: 运行完整测试（268 passed）**

Run: `python -m pytest -q`

Expected: 全部 PASS。

- [x] **Step 3: 检查差异与临时文件**

Run: `git diff --check`

Expected: 无空白错误；确认 `outputs11/`、用户默认视频路径和无关工作区改动未被修改或提交。

- [x] **Step 4: 标记计划完成**

将本计划所有复选框更新为 `[x]`，记录实际测试数量，不执行提交或推送。
