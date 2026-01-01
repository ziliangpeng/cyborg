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

### Industry Consensus

Our finding of 256 as optimal aligns with NVIDIA's official guidance and community best practices:

- **NVIDIA CUDA Programming Guide**: "A thread block size of 16×16 (256 threads) is a common choice"
- **Common practice**: Industry standard values are 128, 256, or 512 threads/block
- **Warp alignment**: 256 = 8 warps × 32 threads, perfect for hardware utilization
- **Benchmarks**: Studies show 256 provides best balance of occupancy and resource usage

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

## Experiment 2: Kernel Fusion Impact (VMA)

**Question:** Does fusing multiply-add into a single kernel improve performance vs running separate kernels?

**Setup:**
- Array sizes: 1K, 10K, 100K, 1M, 10M, 100M elements
- Block size: 256 threads/block (optimal from Experiment 1)
- Operation: Vector Multiply-Add (VMA): `d[i] = a[i] * b[i] + c[i]`
- Iterations: 1000 runs per configuration
- Metric: Median execution time

**Comparison:**
- **Fused**: Single `vectorFMA` kernel
- **Separate**: `vectorMul` then `vectorAdd` (no sync between - kernels auto-order)

### Results

| Array Size | Separate (mul+add) | Fused (FMA) | Speedup |
|------------|-------------------|-------------|---------|
| 1K | 0.009 ms | 0.006 ms | 1.5x |
| 10K | 0.009 ms | 0.006 ms | 1.5x |
| 100K | 0.009 ms | 0.006 ms | 1.5x |
| 1M | 0.013 ms | 0.008 ms | 1.6x |
| 10M | 0.094 ms | 0.058 ms | 1.6x |
| 100M | 0.861 ms | 0.527 ms | 1.6x |

### Key Findings

1. **Consistent 1.5-1.6x speedup from fusion**
   - Benefit is uniform across all workload sizes
   - Fusion always wins

2. **Small workloads (≤100K): 1.5x speedup**
   - Double kernel launch overhead (separate has 2 launches vs 1)
   - Minimal computation time (~0.009 ms baseline for separate)
   - Fused eliminates one launch

3. **Large workloads (≥1M): 1.6x speedup**
   - Memory traffic dominates
   - Separate must write intermediate result (temp) to global memory
   - Fused keeps intermediate value in registers
   - Saves bandwidth: no temp buffer write/read

4. **No sync needed between kernels**
   - Initially tested with `cudaDeviceSynchronize()` between mul and add
   - Removed it - kernels on same stream auto-order
   - Separate mode improved ~1.6x from removing unnecessary sync

### Why Fusion Wins

**Fused kernel benefits:**
- ✅ Single kernel launch (vs 2 launches)
- ✅ Intermediate value stays in registers (no memory traffic)
- ✅ Better instruction pipelining
- ✅ Lower scheduling overhead

**Separate kernel overhead:**
- ❌ Two kernel launches per iteration
- ❌ Must write temp result to global memory
- ❌ Must read temp result back for second kernel
- ❌ Double the memory bandwidth usage

### Performance Analysis

For 100M elements:
- Fused: 0.527 ms
- Separate: 0.861 ms
- Extra cost: 0.334 ms for intermediate memory traffic

**Memory traffic calculation:**
- Temp buffer: 100M × 4 bytes = 400 MB
- Write + Read = 800 MB total
- Time: 0.334 ms
- Implied bandwidth: 800 MB / 0.334 ms = **2.4 GB/s**

This is well below peak bandwidth, suggesting the overhead comes from:
- Kernel launch latency (~10-20 μs each)
- Memory access latency
- Scheduler overhead

### Recommendations

1. **Always fuse operations when possible**
   - Consistent 1.5-1.6x speedup
   - No downsides

2. **Kernel fusion is critical for:**
   - Complex operations with intermediate results
   - Compute pipelines (multiply, add, divide chains)
   - Image/signal processing filters

3. **Don't use cudaDeviceSynchronize() between kernels**
   - Kernels on same stream execute in order automatically
   - Sync only when CPU needs to access results

---

*Last Updated: 2025-12-31*
