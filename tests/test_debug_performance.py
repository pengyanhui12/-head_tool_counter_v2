import time

from core.debug_events import DebugStats, PerfTimer, TimedMatcher


def test_perf_timer_records_calls_and_average():
    stats = DebugStats()

    with PerfTimer(stats, "quality_ms"):
        time.sleep(0.001)
    with PerfTimer(stats, "quality_ms"):
        time.sleep(0.001)

    row = next(row for row in stats.performance_rows() if row[0] == "quality_ms")
    assert row[2] == 2
    assert row[3] == row[1] / 2


def test_timed_matcher_delegates_and_records_feature_match():
    class FakeMatcher:
        def match(self, source, target):
            return (source, target)

    stats = DebugStats()
    matcher = TimedMatcher(FakeMatcher(), stats)

    assert matcher.match("a", "b") == ("a", "b")
    assert stats.feature_match_ms >= 0
    assert stats.timing_calls["feature_match_ms"] == 1


def test_performance_rows_report_unaccounted_wall_time():
    stats = DebugStats()
    stats.add_timing("quality_ms", 20.0)
    stats.add_timing("debug_image_write_ms", 10.0)

    assert stats.accounted_ms == 30.0
    assert stats.unaccounted_ms(50.0) == 20.0
    assert stats.coverage_ratio(50.0) == 0.6


def test_l2_result_counter_tracks_actual_runs_and_empty_results():
    stats = DebugStats()

    stats.record_l2_result(3)
    stats.record_l2_result(0)

    assert stats.l2_run_frames == 2
    assert stats.l2_empty_frames == 1


def test_l2_counter_consistency_uses_inference_call_count():
    stats = DebugStats()
    stats.add_timing("l2_inference_ms", 5.0)
    stats.record_l2_result(1)

    assert stats.l2_run_count_inconsistency == 0

    stats.record_l2_result(1)
    assert stats.l2_run_count_inconsistency == 1
