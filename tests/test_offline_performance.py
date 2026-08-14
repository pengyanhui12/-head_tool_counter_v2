from core.debug_events import DebugStats
from apps.offline_scan import print_performance_stats, resolve_performance_enabled


def test_performance_override_takes_precedence_over_config():
    assert resolve_performance_enabled({"enable_performance_stats": False}, True)
    assert not resolve_performance_enabled({"enable_performance_stats": True}, False)
    assert resolve_performance_enabled({"enable_performance_stats": True}, None)


def test_performance_table_is_printed_when_called(capsys):
    stats = DebugStats()
    stats.add_timing("quality_ms", 12.0)

    print_performance_stats(stats, wall_ms=20.0, total_frames=2)

    output = capsys.readouterr().out
    assert "Performance breakdown" in output
    assert "Quality evaluation" in output
    assert "Accounted coverage" in output
