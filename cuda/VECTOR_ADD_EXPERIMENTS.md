# Vector Addition Block Size Experiments

**Environment:** H100 80GB, PCIe Gen4 x16, CUDA 12.4, sm_90 architecture

---

## Experiment 1: Block Size Impact on Performance

**Question:** How does thread block size affect kernel performance across different array sizes?

**Setup:**
- Array sizes: 10K, 100K, 1M, 10M, 100M elements
- Block sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 threads/block
- Kernel: Simple vector addition (c[i] = a[i] + b[i])
- Iterations: 1000 runs per configuration
- Metric: Median execution time

### Results

| Block Size | 10K | 100K | 1M | 10M | 100M |
|------------|-----|------|-----|------|--------|
| **1** | 0.012 ms | 0.066 ms | 0.607 ms | 6.015 ms | 60.097 ms |
| **2** | 0.009 ms | 0.036 ms | 0.307 ms | 3.011 ms | 30.054 ms |
| **4** | 0.007 ms | 0.021 ms | 0.156 ms | 1.508 ms | 15.029 ms |
| **8** | 0.007 ms | 0.014 ms | 0.081 ms | 0.757 ms | 7.517 ms |
| **16** | 0.006 ms | 0.010 ms | 0.043 ms | 0.382 ms | 3.762 ms |
| **32** | 0.006 ms | 0.008 ms | 0.025 ms | 0.194 ms | 1.884 ms |
| **64** | 0.006 ms | 0.007 ms | 0.015 ms | 0.100 ms | 0.945 ms |
| **128** | 0.006 ms | 0.007 ms | 0.011 ms | 0.053 ms | 0.479 ms |
| **256** | **0.006 ms** | **0.006 ms** | **0.008 ms** | **0.049 ms** | **0.432 ms** |
| **512** | 0.006 ms | 0.006 ms | 0.008 ms | 0.050 ms | 0.444 ms |
| **1024** | 0.006 ms | 0.006 ms | 0.008 ms | 0.053 ms | 0.469 ms |

**Speedup from block=1 to block=256:**

| Array Size | Speedup | Time Reduction |
|------------|---------|----------------|
| 10K | 2x | 0.012 → 0.006 ms |
| 100K | 11x | 0.066 → 0.006 ms |
| 1M | 76x | 0.607 → 0.008 ms |
| 10M | 123x | 6.015 → 0.049 ms |
| 100M | 139x | 60.097 → 0.432 ms |

### Key Findings

1. **Optimal block size: 256 threads**
   - Best performance across all array sizes
   - Consistent sweet spot from 10K to 100M elements

2. **Diminishing returns after 256**
   - 512 and 1024 show no improvement or slight degradation
   - Likely due to register pressure and resource contention

3. **Performance scales with block size (up to 256)**
   - Each doubling of block size roughly halves execution time
   - Relationship breaks down beyond 256 threads/block

4. **Small workload characteristics (≤100K elements)**
   - Launch overhead dominates execution time
   - Performance saturates quickly (by block=256)
   - Block size less critical

5. **Large workload characteristics (≥1M elements)**
   - Block size becomes critical for performance
   - Dramatic speedup potential (76-139x)
   - More sensitive to suboptimal block sizes

### Analysis

**Why 256 threads/block is optimal:**

- **H100 SM architecture**: Efficiently schedules warps (32 threads each)
  - 256 threads = 8 warps per block
  - Good balance for SM occupancy

- **Resource utilization**: Avoids excessive register/shared memory pressure
  - Allows multiple blocks per SM
  - Better hiding of memory latency

- **Grid dimensions**: Creates sufficient parallelism
  - Example: 10M elements, 256 threads/block = 39,063 blocks
  - Well-distributed across 132 SMs on H100

**Why small block sizes are slow:**

- **Too many blocks**: Excessive scheduling overhead
  - Example: 100M elements, 1 thread/block = 100M blocks!
  - Scheduler becomes bottleneck

- **Underutilized warps**: Threads in a warp execute in lockstep
  - Block size 1-31: Only partial warp utilization
  - Wasting GPU parallelism

**Why large block sizes (>256) don't help:**

- **Resource contention**: More threads compete for SM resources
- **No additional parallelism**: Work is already well-distributed
- **Register pressure**: May spill to local memory

### Recommendations

1. **Use 256 threads/block as default** for memory-bound kernels like vector addition

2. **Consider problem size:**
   - Small (≤100K): Block size less critical, use 256 for consistency
   - Large (≥1M): Block size crucial, 256 provides best performance

3. **Experiment for compute-intensive kernels:**
   - This experiment focused on memory-bound operations
   - Compute-intensive kernels may have different optimal points
   - Always profile with your specific workload

4. **Multiple of 32 (warp size):**
   - Always use block sizes that are multiples of 32
   - Aligns with GPU's warp-based execution model

---

*Last Updated: 2025-12-31*
