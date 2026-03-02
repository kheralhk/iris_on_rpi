# profiling.py

import os
import sys
import time
import functools
import atexit
from contextlib import contextmanager
from collections import defaultdict

PROFILE_ENABLED = os.environ.get("IRIS_PROFILE", "0") == "1"

_totals = defaultdict(lambda: {"count": 0, "total": 0.0})

def _log_timing(name: str, dt_seconds: float):
    print(f"[timing] {name}: {dt_seconds:.5f} seconds", file=sys.stderr)

def _accumulate(name: str, dt_seconds: float):
    d = _totals[name]
    d["count"] += 1
    d["total"] += dt_seconds

@atexit.register
def _print_summary():
    if not PROFILE_ENABLED or not _totals:
        return
    print("[timing-summary] totals:", file=sys.stderr)
    # Sort by total time descending
    items = sorted(_totals.items(), key=lambda kv: kv[1]["total"], reverse=True)
    for name, stats in items:
        print(f"[timing-summary] {name}: total={stats['total']:.5f}s, calls={stats['count']}", file=sys.stderr)

def timeit(func):
    if not PROFILE_ENABLED:
        return func
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        _accumulate(func.__qualname__, dt)
        return result
    return wrapper

@contextmanager
def span(name: str):
    if not PROFILE_ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _accumulate(name, dt)
