import time
from functools import wraps
from collections import defaultdict
import threading

# TODO: Move this profiler module to a separate library for reuse across projects

PROFILING_ENABLED = False
_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'times': []})
_stats_lock = threading.Lock()

def enable_profiling():
    global PROFILING_ENABLED
    PROFILING_ENABLED = True

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not PROFILING_ENABLED:
            return func(*args, **kwargs)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        with _stats_lock:
            _stats[func.__name__]['count'] += 1
            _stats[func.__name__]['total_time'] += elapsed
            _stats[func.__name__]['times'].append(elapsed)

        return result
    return wrapper

def _percentile(data, percentile):
    if not data:
        return 0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]

def print_stats():
    if not PROFILING_ENABLED or not _stats:
        return

    print("\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)

    for func_name, data in sorted(_stats.items()):
        times = data['times']
        count = data['count']
        total = data['total_time']

        avg = total / count
        min_time = min(times)
        max_time = max(times)
        p50 = _percentile(times, 50)
        p90 = _percentile(times, 90)
        p95 = _percentile(times, 95)
        p99 = _percentile(times, 99)

        print(f"\n{func_name}:")
        print(f"  Calls: {count}")
        print(f"  Total: {total:.2f}s")
        print(f"  Avg: {avg*1000:.2f}ms")
        print(f"  Min: {min_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")
        print(f"  P50: {p50*1000:.2f}ms")
        print(f"  P90: {p90*1000:.2f}ms")
        print(f"  P95: {p95*1000:.2f}ms")
        print(f"  P99: {p99*1000:.2f}ms")
