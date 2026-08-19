from core.performance_profiler import (
    PerformanceProfiler,
    resolve_performance_enabled,
)


def test_performance_override_takes_precedence_over_config():
    assert resolve_performance_enabled({"enable_performance_stats": False}, True)
    assert not resolve_performance_enabled({"enable_performance_stats": True}, False)
    assert resolve_performance_enabled({"enable_performance_stats": True}, None)


def test_performance_table_uses_pipeline_wall_time_label():
    profiler = PerformanceProfiler(enabled=True, started_at=1.0)
    profiler.record("quality", 12.0)

    output = profiler.format_report(
        profiler.snapshot(total_frames=2, ended_at=1.02)
    )

    assert "Performance breakdown" in output
    assert "Quality evaluation" in output
    assert "Accounted coverage" in output
    assert "Pipeline wall time" in output
    assert "End-to-end" not in output
