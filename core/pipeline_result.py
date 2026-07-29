"""Pipeline 运行结果容器 — 统一收集所有输出指标"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.debug_events import DebugStats


@dataclass
class PipelineResult:
    """一次完整 pipeline 运行的所有输出。"""

    run_id: str
    video_path: str
    git_sha: str = ""

    # ── 统计 ──
    stats: DebugStats = field(default_factory=DebugStats)

    # ── 对象计数 ──
    provisional_objects: int = 0
    confirmed_count: int = 0
    tentative_count: int = 0
    uncertain_count: int = 0
    rejected_count: int = 0
    reportable_count: int = 0

    # ── 关键帧 ──
    total_keyframes: int = 0
    max_consecutive_keyframes: int = 0

    # ── 拒绝原因分布 ──
    rejected_reasons: dict[str, int] = field(default_factory=dict)

    # ── 合并审计 ──
    merge_audits: list = field(default_factory=list)
    total_merges: int = 0
    blocked_merges: int = 0

    # ── recovery ──
    recovery_attempts: int = 0
    recovery_successes: int = 0
    recovery_failures: int = 0

    # ── FPS ──
    core_fps: float = 0.0
    total_fps: float = 0.0

    # ── 输出路径 ──
    output_dir: Path | None = None
    events_path: Path | None = None
    report_json_path: Path | None = None
    report_csv_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "video_path": self.video_path,
            "git_sha": self.git_sha,
            "provisional_objects": self.provisional_objects,
            "confirmed_count": self.confirmed_count,
            "tentative_count": self.tentative_count,
            "uncertain_count": self.uncertain_count,
            "rejected_count": self.rejected_count,
            "reportable_count": self.reportable_count,
            "total_keyframes": self.total_keyframes,
            "max_consecutive_keyframes": self.max_consecutive_keyframes,
            "rejected_reasons": self.rejected_reasons,
            "total_merges": self.total_merges,
            "blocked_merges": self.blocked_merges,
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
            "recovery_failures": self.recovery_failures,
            "core_fps": self.core_fps,
            "total_fps": self.total_fps,
            "track_binding_conflicts": self.stats.track_binding_conflicts,
            "same_frame_duplicate_observations": self.stats.same_frame_duplicate_observations,
            "unexplained_rejected_objects": self.stats.unexplained_rejected_objects,
            "invalid_homography_nodes": self.stats.invalid_homography_nodes,
            "report_count_inconsistency": self.stats.report_count_inconsistency,
        }
