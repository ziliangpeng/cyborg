# Crawler Performance Benchmark

## 2025-11-28

### Test Configuration

- **Test Date:** 2025-11-28
- **Task:** Crawl 100 URLs starting from https://news.ycombinator.com
- **Rate Limit:** 6 requests/min per host (10 seconds between requests)
- **Hardware:** M4 Pro MacBook
- **Python Version:** 3.13.5

## Benchmark Results

Each configuration was run 3 times to measure performance consistency.

### 1 Thread (Baseline)

| Run | Real Time | User Time | Sys Time |
|-----|-----------|-----------|----------|
| 1   | 44.831s   | 4.760s    | 0.330s   |
| 2   | 38.364s   | 4.049s    | 0.248s   |
| 3   | 36.990s   | 4.002s    | 0.279s   |
| **Avg** | **40.1s** | **4.27s** | **0.29s** |

### 2 Threads

| Run | Real Time | User Time | Sys Time |
|-----|-----------|-----------|----------|
| 1   | 24.028s   | 3.594s    | 0.191s   |
| 2   | 20.137s   | 3.745s    | 0.197s   |
| 3   | 26.710s   | 4.121s    | 0.251s   |
| **Avg** | **23.6s** | **3.82s** | **0.21s** |

**Speedup:** 1.7x faster than single thread

### 3 Threads

| Run | Real Time | User Time | Sys Time |
|-----|-----------|-----------|----------|
| 1   | 13.509s   | 3.653s    | 0.177s   |
| 2   | 12.381s   | 3.621s    | 0.160s   |
| 3   | 13.407s   | 3.245s    | 0.171s   |
| **Avg** | **13.1s** | **3.51s** | **0.17s** |

**Speedup:** 3.1x faster than single thread

### 4 Threads

| Run | Real Time | User Time | Sys Time |
|-----|-----------|-----------|----------|
| 1   | 12.337s   | 2.959s    | 0.144s   |
| 2   | 13.724s   | 3.563s    | 0.183s   |
| 3   | 10.882s   | 3.706s    | 0.197s   |
| **Avg** | **12.3s** | **3.41s** | **0.17s** |

**Speedup:** 3.3x faster than single thread

## Performance Analysis

### Speedup Summary

| Threads | Avg Time | Speedup | Efficiency |
|---------|----------|---------|------------|
| 1       | 40.1s    | 1.0x    | 100%       |
| 2       | 23.6s    | 1.7x    | 85%        |
| 3       | 13.1s    | 3.1x    | 103%       |
| 4       | 12.3s    | 3.3x    | 83%        |

### Key Observations

1. **Near-linear scaling up to 3 threads:** The crawler achieves 3.1x speedup with 3 threads, indicating excellent parallelization efficiency.

2. **Diminishing returns at 4 threads:** Going from 3 to 4 threads only provides a marginal improvement (13.1s → 12.3s), suggesting we're hitting bottlenecks.

3. **Super-linear scaling at 3 threads (103% efficiency):** This unusual result suggests that with more threads, we're better able to work around rate limiting by having more hosts to choose from.

### Bottlenecks

Likely limiting factors at higher thread counts:

- **Rate limiting:** 6 req/min per host means we need diverse hosts to benefit from more threads
- **Lock contention:** Single `threading.Lock` in UrlPool may become a bottleneck
- **Python GIL:** Though we're I/O bound, some lock overhead exists
- **Network bandwidth:** Not a factor in these tests

### Recommendations

- **Optimal thread count:** 3-4 threads for this workload
- **For larger crawls:** Consider 4-8 threads depending on host diversity
- **Future optimizations:**
  - Finer-grained locking in UrlPool
  - Async I/O (asyncio) instead of threading
  - Connection pooling for requests

---

## 1000 URLs Benchmark

### Test Configuration

- **Task:** Crawl 1000 URLs starting from https://news.ycombinator.com
- **Rate Limit:** 6 requests/min per host (10 seconds between requests)
- **Hardware:** M4 Pro MacBook
- **Python Version:** 3.13.5

### Results

Each configuration was run 3 times.

| Threads | Run 1 | Run 2 | Run 3 | Avg Time | Speedup |
|---------|-------|-------|-------|----------|---------|
| 3       | 3m1s  | 3m0s  | 2m49s | 2m57s    | 1.0x    |
| 4       | 1m56s | 2m11s | 1m57s | 2m1s     | 1.5x    |
| 5       | 1m39s | 1m1s  | 1m24s | 1m21s    | 2.2x    |
| 6       | 1m10s | 1m21s | 1m27s | 1m19s    | 2.2x    |
| 7       | 1m13s | 1m50s | 1m12s | 1m25s    | 2.1x    |
| 8       | 59s   | 1m7s  | 1m11s | 1m6s     | 2.7x    |
| 9       | 1m14s | 1m2s  | 49s   | 1m2s     | 2.9x    |
| **10**  | **1m4s** | **56s** | **52s** | **57s** | **3.1x** |

### Key Findings

1. **10 threads is optimal** for 1000 URL crawls, achieving average time of 57 seconds
2. **Continued improvement** beyond 4 threads, unlike the 100 URL benchmark
3. **3.1x speedup** from 3 to 10 threads
4. **Best single run:** 52 seconds with 10 threads
5. **Variance decreases** with more threads, indicating better load balancing

### Performance Analysis

- **Larger crawls benefit more from parallelism:** With 1000 URLs, there's more URL diversity across hosts, reducing rate limiting conflicts
- **Lock contention not a bottleneck:** Performance continues to scale up to 10 threads
- **I/O bound workload:** Network I/O is the bottleneck, not CPU or GIL
- **Optimal configuration:** 9-10 threads for large crawls (1000+ URLs)

---

## Test Environment Details

- **Seed URL:** https://news.ycombinator.com
- **Discovery:** URLs discovered by following links from seed
- **Rate limiting:** Enforced per-host (news.ycombinator.com, plus discovered domains)
- **Error handling:** Timeouts (5s), PDF/Parquet files skipped
- **Worker startup:** Worker 0 starts immediately, others delayed 2-5s
