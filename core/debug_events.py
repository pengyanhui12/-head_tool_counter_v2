"""调试事件记录器 — 结构化事件流 + 统计 + 性能计时

每次运行产生 events.jsonl，每个事件包含 event_type、frame_id、payload。
统计计数器和性能计时器独立分离，便于分别输出核心算法 FPS 和含 Debug I/O 的总 FPS。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DebugEvent:
    """单个调试事件。"""
    event_type: str
    frame_id: int | None
    payload: dict = field(default_factory=dict)


class DebugStats:
    """运行期计数器，跟踪所有不变量和性能指标。"""

    def __init__(self):
        # ── 事件计数 ──
        self.l2_run_frames: int = 0
        self.l2_empty_frames: int = 0
        self.keyframe_trigger_quality_drop: int = 0
        self.keyframe_trigger_new_det: int = 0
        self.keyframe_trigger_max_interval: int = 0
        self.keyframe_accepted: int = 0
        self.keyframe_rejected_by_cooldown: int = 0
        self.consecutive_keyframe_max_run: int = 0
        self._consecutive_current: int = 0

        # ── track 不变量 ──
        self.track_binding_conflicts: int = 0
        self.same_frame_duplicate_observations: int = 0

        # ── merge 不变量 ──
        self.merge_blocked_by_cooccurrence: int = 0
        self.merge_blocked_by_frame_overlap: int = 0
        self.objects_created: int = 0
        self.objects_matched_by_track: int = 0
        self.objects_matched_by_cost: int = 0
        self.objects_merged: int = 0
        self.unexplained_rejected_objects: int = 0

        # ── recovery ──
        self.recovery_attempts: int = 0
        self.recovery_success: int = 0
        self.recovery_failed: int = 0

        # ── homography ──
        self.invalid_homography_nodes: int = 0

        # ── report ──
        self.report_count_inconsistency: int = 0

        # ── 性能计时（累计 ms）──
        self.video_decode_ms: float = 0.0
        self.quality_ms: float = 0.0
        self.l2_inference_ms: float = 0.0
        self.tracker_preview_ms: float = 0.0
        self.keyframe_decision_ms: float = 0.0
        self.feature_match_ms: float = 0.0
        self.l1_inference_ms: float = 0.0
        self.l3_inference_ms: float = 0.0
        self.fusion_ms: float = 0.0
        self.tracker_update_ms: float = 0.0
        self.projection_ms: float = 0.0
        self.association_ms: float = 0.0
        self.debug_image_write_ms: float = 0.0
        self.final_review_ms: float = 0.0
        self.report_ms: float = 0.0

        # ── frame计数 ──
        self.total_frames: int = 0

    def record_consecutive_keyframe(self, accepted: bool):
        if accepted:
            self._consecutive_current += 1
            if self._consecutive_current > self.consecutive_keyframe_max_run:
                self.consecutive_keyframe_max_run = self._consecutive_current
        else:
            self._consecutive_current = 0

    @property
    def core_algorithm_ms(self) -> float:
        """核心算法耗时（不含 debug 图片写盘）。"""
        return (self.video_decode_ms + self.quality_ms + self.l2_inference_ms +
                self.tracker_preview_ms + self.keyframe_decision_ms + self.feature_match_ms +
                self.l1_inference_ms + self.l3_inference_ms + self.fusion_ms +
                self.tracker_update_ms + self.projection_ms + self.association_ms +
                self.final_review_ms + self.report_ms)

    @property
    def total_ms(self) -> float:
        """包含 debug 图片写盘的总耗时。"""
        return self.core_algorithm_ms + self.debug_image_write_ms


class DebugEventWriter:
    """将事件写入 events.jsonl，可选静默模式。"""

    def __init__(self, output_path: Path):
        self._path = output_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(str(self._path), "w", encoding="utf-8")

    def emit(self, event: DebugEvent):
        record = {
            "event_type": event.event_type,
            "frame_id": event.frame_id,
            **event.payload,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PerfTimer:
    """上下文管理器，累计某阶段的耗时。"""

    def __init__(self, stats: DebugStats, field_name: str):
        self._stats = stats
        self._field = field_name
        self._t0: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self._t0) * 1000.0  # ms
        current = getattr(self._stats, self._field, 0.0)
        setattr(self._stats, self._field, current + elapsed)
