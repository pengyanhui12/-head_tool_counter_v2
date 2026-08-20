import json

import pytest

from core.performance_profiler import PerformanceProfiler


def test_profiler_accumulates_stage_calls_and_average():
    profiler = PerformanceProfiler(enabled=True, started_at=10.0)

    profiler.record("quality", 12.0)
    profiler.record("quality", 8.0)
    snapshot = profiler.snapshot(total_frames=2, ended_at=10.1)

    quality = snapshot["stages"]["quality"]
    assert quality["total_ms"] == 20.0
    assert quality["calls"] == 2
    assert quality["average_ms"] == 10.0
    assert quality["wall_percent"] == pytest.approx(20.0)


def test_profiler_rejects_unknown_stage():
    profiler = PerformanceProfiler(enabled=True)

    with pytest.raises(ValueError, match="unknown performance stage"):
        profiler.record("misspelled_stage", 1.0)


def test_disabled_profiler_does_not_accumulate_or_save(tmp_path):
    profiler = PerformanceProfiler(enabled=False, started_at=10.0)

    profiler.record("quality", 12.0)
    result = profiler.save(tmp_path, total_frames=1, ended_at=10.1)

    assert result is None
    assert not (tmp_path / "reports" / "performance.json").exists()
    assert not (tmp_path / "reports" / "performance.txt").exists()


def test_failed_operation_does_not_count_as_success():
    profiler = PerformanceProfiler(enabled=True, started_at=10.0)

    with pytest.raises(StopIteration):
        profiler.measure_success("video_decode", lambda: next(iter(())))

    snapshot = profiler.snapshot(total_frames=0, ended_at=10.1)
    assert snapshot["stages"]["video_decode"]["calls"] == 0


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


def test_overlap_is_reported_instead_of_clamped():
    profiler = PerformanceProfiler(enabled=True, started_at=10.0)
    profiler.record("quality", 120.0)

    snapshot = profiler.snapshot(total_frames=1, ended_at=10.1)

    assert snapshot["unaccounted_ms"] == pytest.approx(-20.0)
    assert snapshot["timing_overlap_detected"] is True
    assert snapshot["accounted_coverage"] == pytest.approx(1.2)


def test_report_uses_unambiguous_pipeline_wall_label():
    profiler = PerformanceProfiler(enabled=True, started_at=1.0)
    text = profiler.format_report(
        profiler.snapshot(total_frames=2, ended_at=1.02)
    )

    assert "Pipeline wall time" in text
    assert "End-to-end" not in text


def test_save_writes_machine_and_human_readable_reports(tmp_path):
    profiler = PerformanceProfiler(enabled=True, started_at=10.0)
    profiler.record("quality", 20.0)

    snapshot = profiler.save(tmp_path, total_frames=2, ended_at=10.1)

    json_path = tmp_path / "reports" / "performance.json"
    text_path = tmp_path / "reports" / "performance.txt"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved == snapshot
    assert "Pipeline wall time" in text_path.read_text(encoding="utf-8")


def test_wrapped_matcher_delegates_and_records_feature_match():
    class FakeMatcher:
        marker = "delegated"

        def match(self, source, target):
            return source, target

    profiler = PerformanceProfiler(enabled=True, started_at=1.0)
    matcher = profiler.wrap_matcher(FakeMatcher())

    assert matcher.match("a", "b") == ("a", "b")
    assert matcher.marker == "delegated"
    snapshot = profiler.snapshot(total_frames=1, ended_at=1.1)
    assert snapshot["stages"]["feature_match"]["calls"] == 1
