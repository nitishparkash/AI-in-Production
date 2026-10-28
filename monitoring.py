"""
Performance monitoring module.

Tracks:
  - API request latency
  - Prediction throughput
  - Model load times
  - Memory / system info snapshots

Exposes helpers consumed by the Flask API and a /metrics endpoint.
"""

import os
import time
import json
import psutil
from datetime import datetime, timezone
from collections import defaultdict
from config import get_config, ensure_directories
from logger import get_logger, log_event


class PerformanceMonitor:
    """Lightweight in-process performance tracker."""

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        ensure_directories(cfg)
        self.logger = get_logger("monitoring", cfg)
        self._timings = defaultdict(list)  # endpoint → [durations]
        self._call_counts = defaultdict(int)
        self._start_time = time.time()

    # ── timing context manager ──
    class _Timer:
        def __init__(self, monitor, label):
            self.monitor = monitor
            self.label = label

        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            elapsed = round(time.time() - self.t0, 6)
            self.monitor._timings[self.label].append(elapsed)
            self.monitor._call_counts[self.label] += 1
            log_event(self.monitor.logger, "MONITOR",
                      f"{self.label} completed", duration_s=elapsed)

    def time(self, label):
        """Return a context manager that records wall-clock duration.

        Usage
        -----
        >>> with monitor.time("predict"):
        ...     result = model.predict(X)
        """
        return self._Timer(self, label)

    # ── metrics ──
    def get_metrics(self):
        """Return a JSON-serializable metrics summary."""
        uptime = round(time.time() - self._start_time, 2)
        process = psutil.Process(os.getpid())
        mem = process.memory_info()

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_rss_mb": round(mem.rss / 1e6, 2),
                "memory_vms_mb": round(mem.vms / 1e6, 2),
            },
            "endpoints": {},
        }

        for label, durations in self._timings.items():
            metrics["endpoints"][label] = {
                "calls": self._call_counts[label],
                "avg_latency_s": round(sum(durations) / len(durations), 6),
                "max_latency_s": round(max(durations), 6),
                "min_latency_s": round(min(durations), 6),
                "total_time_s": round(sum(durations), 4),
            }

        return metrics

    def save_snapshot(self):
        """Persist current metrics snapshot to the monitoring log file."""
        metrics = self.get_metrics()
        with open(self.cfg.MONITORING_LOG, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        log_event(self.logger, "MONITOR", "Metrics snapshot saved",
                  path=self.cfg.MONITORING_LOG)
        return metrics
