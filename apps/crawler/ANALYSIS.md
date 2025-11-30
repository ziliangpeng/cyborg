# Crawler Performance Analysis

## 2025-11-30

### Test Configuration

- **Date:** 2025-11-30
- **Task:** Crawl 5000 URLs starting from https://news.ycombinator.com
- **Rate Limit:** 6 requests/min per host (10 seconds between requests)
- **Implementation:** Async crawler with aiohttp
- **Hardware:** M4 Pro MacBook
- **Python Version:** 3.13.5
- **Connection Pool:** 5000 max connections, 2 per host
- **DNS Cache:** 300 seconds TTL

## Overall Throughput Summary

| Workers | Real Time (s) | Throughput (URLs/s) | Speedup vs 5 workers |
|---------|---------------|---------------------|----------------------|
| 5       | 394.84        | 12.66               | 1.00x                |
| 10      | 211.24        | 23.67               | 1.87x                |
| 15      | 164.42        | 30.41               | 2.40x                |
| 20      | 136.93        | 36.51               | 2.88x                |
| 30      | 142.12        | 35.19               | 2.78x                |
| 40      | 125.29        | 39.91               | 3.15x                |
| 50      | 124.93        | 40.02               | 3.16x                |
| **75**  | **114.95**    | **43.50**           | **3.44x**            |
| 100     | 122.83        | 40.71               | 3.22x                |

### Key Findings

1. **Optimal Worker Count: 75 workers**
   - Best performance at 114.95s (43.50 URLs/s)
   - 100 workers actually performs slightly worse (122.83s)
   - Diminishing returns after 40-50 workers

2. **Scaling Characteristics**
   - Near-linear scaling up to 20 workers (2.88x speedup)
   - Good scaling up to 40-50 workers (3.15x speedup)
   - Performance plateau/slight regression at 100 workers

## Component Performance Analysis

### extract_links (CPU-bound HTML Parsing)

| Workers | Avg Time | P90 Time | P99 Time | Total Time |
|---------|----------|----------|----------|------------|
| 5       | 14.79ms  | 31.16ms  | 117.97ms | 61.32s     |
| 10      | 11.84ms  | 24.84ms  | 95.86ms  | 51.63s     |
| 20      | 11.61ms  | 25.06ms  | 107.60ms | 49.58s     |
| 30      | 13.65ms  | 26.88ms  | 151.55ms | 59.58s     |
| 40      | 12.82ms  | 27.97ms  | 128.63ms | 53.46s     |
| 50      | 14.97ms  | 35.50ms  | 130.86ms | 61.25s     |
| 75      | 13.25ms  | 30.13ms  | 137.76ms | 52.82s     |
| 100     | 11.50ms  | 25.14ms  | 117.54ms | 44.83s     |

**Analysis:**
- ✅ **Extract_links is NOT a bottleneck**
- Average time stays stable at 11-15ms across all worker counts
- Thread pool executor successfully offloads CPU work without blocking the event loop
- Total time stays relatively constant (45-61s) despite varying worker counts

### fetch (Network I/O)

| Workers | Avg Time | P50 Time | P90 Time | P99 Time | Total Time |
|---------|----------|----------|----------|----------|------------|
| 5       | 370ms    | 189ms    | 805ms    | 3260ms   | 1849s      |
| 10      | 389ms    | 224ms    | 820ms    | 2848ms   | 1944s      |
| 15      | 448ms    | 264ms    | 973ms    | 3641ms   | 2235s      |
| 20      | 485ms    | 305ms    | 955ms    | 5050ms   | 2415s      |
| 30      | 734ms    | 499ms    | 1517ms   | 5310ms   | 3649s      |
| 40      | 867ms    | 608ms    | 1809ms   | 5257ms   | 4299s      |
| 50      | 1085ms   | 713ms    | 2398ms   | 5513ms   | 5369s      |
| 75      | 1502ms   | 990ms    | 3326ms   | 6057ms   | 7353s      |
| 100     | 2109ms   | 1353ms   | 4917ms   | 6632ms   | 10337s     |

**Critical Finding:** 🔴 **Fetch time INCREASES dramatically with more workers!**

- At 5 workers: avg 370ms, P50 189ms
- At 100 workers: avg 2109ms (5.7x slower!), P50 1353ms (7.2x slower!)
- Total fetch time increases from 1849s → 10337s (5.6x increase)
- P99 latency approaches timeout limit (6.6s vs 5s timeout)

## Why Does fetch Get Slower?

### Rate Limiting Contention

With more concurrent workers competing for the same limited pool of hosts:

1. **Rate limit enforcement:** 6 requests/min per host = 10 second wait between requests
2. **More workers = more contention:** Multiple workers want to fetch from the same hosts
3. **Increased wait times:** Workers spend more time waiting for rate limits to expire
4. **Individual operations slow down:** Each fetch takes longer on average due to waiting

### Evidence of Rate Limiting Impact

**Wall Clock Time vs Total Function Time:**

At 100 workers:
- Total fetch time: 10337s
- Real wall time: 123s
- **Parallelism factor: 84x** (10337/123)
- On average, 84 fetch operations happening concurrently
- But each operation takes 2109ms average due to rate limit waiting

At 5 workers:
- Total fetch time: 1849s
- Real wall time: 395s
- **Parallelism factor: 4.7x** (1849/395)
- Lower concurrency, but each fetch completes faster (370ms avg)

**Interpretation:**
- Higher worker counts achieve better parallelism
- Overall throughput improves (395s → 123s wall clock)
- But individual fetch operations take longer due to rate limiting
- This is the expected trade-off with per-host rate limiting

## Bottleneck Summary

### 1. Primary Bottleneck: Rate Limiting ⚠️

**Impact:** HIGH
- Per-host rate limit (10s between requests) creates contention at scale
- More workers = more waiting for hosts to become available
- Individual fetch times increase from 370ms → 2109ms avg
- But overall throughput still improves due to parallelism

### 2. Extract Links: NOT a bottleneck ✅

**Impact:** MINIMAL
- Stays stable at ~11-15ms average across all worker counts
- Thread pool executor successfully offloads CPU work from event loop
- Total parsing time remains relatively constant (~45-61s)

### 3. Connection Pool: NOT a bottleneck ✅

**Impact:** MINIMAL
- 5000 connection limit is sufficient for all tested worker counts
- No evidence of connection exhaustion in the metrics
- DNS caching (300s TTL) working effectively

## Performance Sweet Spots

### For 5000 URL workload:

1. **Best overall performance: 75 workers**
   - 114.95s (43.50 URLs/s)
   - 3.44x speedup over 5 workers
   - Good balance between parallelism and rate limit contention

2. **Good performance range: 40-75 workers**
   - 40 workers: 125.29s (39.91 URLs/s)
   - 50 workers: 124.93s (40.02 URLs/s)
   - 75 workers: 114.95s (43.50 URLs/s)

3. **Diminishing returns: 100+ workers**
   - 100 workers: 122.83s (40.71 URLs/s)
   - Slightly worse than 75 workers
   - Rate limiting contention outweighs parallelism gains

## Recommendations

### For Current Implementation

1. **Use 50-75 workers for optimal performance**
   - Best balance of throughput and resource usage
   - 75 workers provides best results for 5000 URL workloads

2. **Avoid 100+ workers**
   - Marginal gains or even performance regression
   - Increased resource usage without benefit

### Future Optimizations

1. **Reduce rate limit strictness (if acceptable):**
   - Current: 6 req/min (10s between requests)
   - Consider: 10-12 req/min (5-6s between requests)
   - Would reduce contention and improve individual fetch times

2. **Implement smarter host selection:**
   - Prioritize hosts that haven't been accessed recently
   - Load balance across discovered domains
   - Could reduce rate limit wait times

3. **Host diversity analysis:**
   - Profile which hosts dominate the crawl
   - If crawling is too concentrated on few hosts, rate limiting will always be an issue
   - Consider breadth-first crawling to discover more hosts earlier

4. **Adaptive worker scaling:**
   - Start with fewer workers, scale up as more diverse hosts are discovered
   - Could optimize early crawl phase vs later phases

## Comparison with Previous Results

### 1000 URLs Benchmark (from BENCHMARK.md)

Previous best (threading):
- 10 threads: 57s (17.5 URLs/s)

Previous best (async):
- 20 workers: 28.41s (35.21 URLs/s)

Current (5000 URLs, async):
- 75 workers: 114.95s (43.50 URLs/s)

**Observation:**
- Throughput continues to improve with scale
- 5000 URL workload achieves higher throughput (43.50 vs 35.21 URLs/s)
- Larger workloads benefit from higher worker counts
- Rate limiting becomes more significant at scale but still allows good parallelism

## Conclusion

The async crawler demonstrates excellent scaling characteristics up to 75 workers for a 5000 URL workload, achieving 3.44x speedup over the baseline. The primary bottleneck is rate limiting contention, which causes individual fetch operations to slow down as worker count increases, but overall throughput continues to improve due to better parallelism. The sweet spot for this workload is 50-75 workers, with 75 workers providing the best performance at 43.50 URLs/s.

Key takeaway: The crawler architecture is sound, with CPU-bound parsing properly offloaded to thread pool. Further improvements would require either relaxing rate limits or implementing smarter host selection strategies to reduce contention.
