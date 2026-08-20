"""离线流水线性能统计、展示与持久化。"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar


T = TypeVar("T")


STAGE_LABELS: dict[str, str] = {
    "initialization": "Initialization",
    "video_decode": "Video decode",
    "quality": "Quality evaluation",
    "l2_inference": "L2 inference",
    "tracker_preview": "Tracker preview",
    "tracker_update": "Tracker update",
    "keyframe_decision": "Keyframe decision",
    "feature_match": "Feature matching",
    "recovery": "Recovery",
    "l1_inference": "L1 inference",
    "l3_inference": "L3 inference",
    "fusion": "Fusion",
    "graph_update": "Graph update",
    "projection": "Projection",
    "association": "Association",
    "coverage_update": "Coverage update",
    "end_window": "End-window processing",
    "final_review": "Final review",
    "mosaic": "Global mosaic",
    "evidence": "Evidence extraction",
    "session_store": "Session storage",
    "report": "Report generation",
    "event_log_io": "Event/log I/O",
    "debug_image_write": "Debug image I/O",
}


def resolve_performance_enabled(
    config: dict,
    override: bool | None,
) -> bool:
    """命令行显式值优先，否则读取流水线配置。"""
    if override is not None:
        return bool(override)
    return bool(config.get("enable_performance_stats", False))


class _TimedMatcher:
    """透明代理匹配器，并把每次 match 归入特征匹配阶段。"""

    def __init__(self, matcher, profiler: "PerformanceProfiler"):
        self._matcher = matcher
        self._profiler = profiler

    def match(self, source, target):
        with self._profiler.measure("feature_match"):
            return self._matcher.match(source, target)

    def __getattr__(self, name):
        return getattr(self._matcher, name)


class PerformanceProfiler:
    """集中管理流水线阶段计时及稳定的性能输出契约。"""

    def __init__(
        self,
        enabled: bool,
        started_at: float | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.started_at = (
            time.perf_counter() if started_at is None else float(started_at)
        )
        self._totals = {stage: 0.0 for stage in STAGE_LABELS}
        self._calls = {stage: 0 for stage in STAGE_LABELS}

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in STAGE_LABELS:
            raise ValueError(f"unknown performance stage: {stage}")

    def record(self, stage: str, elapsed_ms: float) -> None:
        """累计一次已完成阶段；关闭统计时仅保留阶段名校验。"""
        self._validate_stage(stage)
        if not self.enabled:
            return
        self._totals[stage] += float(elapsed_ms)
        self._calls[stage] += 1

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """测量一个阶段；异常退出也记录已经消耗的处理时间。"""
        self._validate_stage(stage)
        if not self.enabled:
            yield
            return
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - started_at) * 1000.0)

    def measure_success(
        self,
        stage: str,
        operation: Callable[[], T],
    ) -> T:
        """只统计成功操作，避免把 EOF 探测记成一次视频解码。"""
        self._validate_stage(stage)
        if not self.enabled:
            return operation()
        started_at = time.perf_counter()
        result = operation()
        self.record(stage, (time.perf_counter() - started_at) * 1000.0)
        return result

    def stage_total(self, stage: str) -> float:
        self._validate_stage(stage)
        return self._totals[stage] if self.enabled else 0.0

    def record_exclusive(
        self,
        stage: str,
        started_at: float,
        excluded_stage: str,
        excluded_before_ms: float,
        ended_at: float | None = None,
    ) -> None:
        """记录扣除嵌套阶段后的独占耗时，防止墙钟时间重复累计。"""
        self._validate_stage(stage)
        self._validate_stage(excluded_stage)
        if not self.enabled:
            return
        finished_at = time.perf_counter() if ended_at is None else ended_at
        wall_ms = (finished_at - started_at) * 1000.0
        nested_ms = self.stage_total(excluded_stage) - excluded_before_ms
        self.record(stage, max(0.0, wall_ms - nested_ms))

    def wrap_matcher(self, matcher):
        """关闭统计时直接返回原对象，避免引入无意义代理层。"""
        return _TimedMatcher(matcher, self) if self.enabled else matcher

    def snapshot(
        self,
        total_frames: int,
        ended_at: float | None = None,
    ) -> dict:
        """冻结一次性能快照；覆盖率不截断，以暴露重复计时。"""
        finished_at = time.perf_counter() if ended_at is None else ended_at
        wall_ms = max(0.0, (finished_at - self.started_at) * 1000.0)
        accounted_ms = sum(self._totals.values()) if self.enabled else 0.0
        unaccounted_ms = wall_ms - accounted_ms
        coverage = accounted_ms / wall_ms if wall_ms > 0 else 0.0
        fps = total_frames / (wall_ms / 1000.0) if wall_ms > 0 else 0.0

        stages = {}
        for stage, label in STAGE_LABELS.items():
            total = self._totals[stage] if self.enabled else 0.0
            calls = self._calls[stage] if self.enabled else 0
            stages[stage] = {
                "label": label,
                "total_ms": total,
                "calls": calls,
                "average_ms": total / calls if calls else 0.0,
                "wall_percent": total / wall_ms * 100.0 if wall_ms > 0 else 0.0,
            }

        return {
            "schema_version": "1.0",
            "total_frames": int(total_frames),
            "pipeline_wall_ms": wall_ms,
            "pipeline_fps": fps,
            "accounted_ms": accounted_ms,
            "unaccounted_ms": unaccounted_ms,
            "accounted_coverage": coverage,
            "timing_overlap_detected": unaccounted_ms < 0.0,
            "stages": stages,
        }

    def format_report(self, snapshot: dict) -> str:
        """生成控制台和 TXT 共用的性能表格。"""
        wall_ms = float(snapshot["pipeline_wall_ms"])
        fps = float(snapshot["pipeline_fps"])
        lines = [
            "Performance breakdown:",
            f"  Pipeline wall time: {wall_ms / 1000.0:.3f}s ({fps:.1f} FPS)",
            (
                f"  {'Stage':<24} {'Total(ms)':>10} {'Calls':>8} "
                f"{'Avg(ms)':>10} {'Wall%':>8}"
            ),
        ]
        for stage in STAGE_LABELS:
            row = snapshot["stages"][stage]
            lines.append(
                f"  {row['label']:<24} {row['total_ms']:>10.1f} "
                f"{row['calls']:>8d} {row['average_ms']:>10.3f} "
                f"{row['wall_percent']:>7.1f}%"
            )
        lines.append(
            f"  {'Unaccounted overhead':<24} "
            f"{snapshot['unaccounted_ms']:>10.1f} {'-':>8} {'-':>10} "
            f"{(snapshot['unaccounted_ms'] / wall_ms * 100.0 if wall_ms else 0.0):>7.1f}%"
        )
        lines.append(
            f"  Accounted coverage: "
            f"{snapshot['accounted_coverage'] * 100.0:.1f}%"
        )
        if snapshot["timing_overlap_detected"]:
            lines.append(
                "  WARNING: timing overlap detected; stage totals exceed "
                "pipeline wall time."
            )
        return "\n".join(lines)

    def save(
        self,
        output_dir: str | Path,
        total_frames: int,
        ended_at: float | None = None,
    ) -> dict | None:
        """保存同一快照的 JSON 与 TXT，避免两个文件出现口径漂移。"""
        if not self.enabled:
            return None
        snapshot = self.snapshot(total_frames=total_frames, ended_at=ended_at)
        reports_dir = Path(output_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "performance.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (reports_dir / "performance.txt").write_text(
            self.format_report(snapshot) + "\n",
            encoding="utf-8",
        )
        return snapshot
