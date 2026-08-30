from preprocessing.performance import PerformanceMonitor


def test_performance_monitor_collects_timing():
    monitor = PerformanceMonitor()
    with monitor.measure("pose"):
        sum(range(100))
    assert monitor.summary()["pose_ms"] >= 0
