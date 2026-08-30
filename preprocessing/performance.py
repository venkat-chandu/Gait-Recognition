"""Lightweight timing and resource-monitoring helpers."""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
import os
import tracemalloc


class PerformanceMonitor:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    @contextmanager
    def measure(self, name: str):
        started = perf_counter()
        yield
        self.samples.setdefault(name, []).append((perf_counter() - started) * 1000)

    def summary(self) -> dict[str, float]:
        report = {f"{name}_ms": sum(items) / len(items) for name, items in self.samples.items() if items}
        report["memory_mb"] = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
        report["pid"] = float(os.getpid())
        return report
