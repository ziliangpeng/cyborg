# Softmax Experiments

**Environment:** H100 80GB, PCIe Gen4 x16, CUDA 12.4, sm_90 architecture

---

## Experiment 1: Numerical Stability - Naive vs Stable Softmax

**Question:** Why is numerical stability critical in softmax computation?

**Setup:**
- Array size: 1,000 elements
- Test cases:
  1. Safe input range: [0, 10]
  2. Overflow input range: [500, 1000]
- Methods tested: Naive (unstable), Multi-pass (stable)

### Results

| Test Case | Method | Input Range | Sum(output) | Has NaN/Inf |
|-----------|--------|-------------|-------------|-------------|
| 1 | Naive | [0, 10] | 1.000000 | NO |
| 2 | Naive | [500, 1000] | nan | **YES (OVERFLOW!)** |
| 3 | Multi-pass | [500, 1000] | 1.000000 | NO |

### Key Findings

1. **Naive softmax fails catastrophically with large inputs**
   - Input value: 750
   - exp(750) ≈ 1.4 × 10^325 → **OVERFLOW to infinity**
   - Result: All outputs become NaN (0/0 or inf/inf)

2. **Max subtraction prevents overflow**
   - Multi-pass: First find max = 1000
   - Compute: exp(750 - 1000) = exp(-250) ≈ 3.7 × 10^-109
   - All exp values in range [0, 1], no overflow!

3. **Why this matters in practice:**
   - **Transformers:** Attention logits can be large (50-100+)
   - **Temperature scaling:** High temperature → large logits
   - **Numerical precision:** Even moderate values (> 88) overflow float32

### The Overflow Problem

**Naive formula:**
```
softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))
```

**Why it fails:**
- `exp(x)` grows exponentially: exp(100) ≈ 2.7 × 10^43
- Float32 max: ~3.4 × 10^38
- **exp(89) already overflows!**

**Stable formula:**
```
m = max(x)
softmax(x)[i] = exp(x[i] - m) / sum(exp(x[j] - m))
```

**Why it works:**
- Largest value: exp(max - max) = exp(0) = 1
- All other values: exp(x - max) < 1 (since x ≤ max)
- **No overflow possible!**

### Demonstration

```bash
# Safe inputs - naive works
$ ./softmax -n 1000 --method naive -v
✓ Verification PASSED

# Large inputs - naive fails
$ # (with input [500, 1000])
✗ Verification FAILED: Output contains NaN or Inf

# Large inputs - multi-pass stable
$ ./softmax -n 1000 --method multi -v
✓ Verification PASSED
```

---

## Experiment 2: Performance Comparison - Naive vs Multi-pass

**Question:** What is the performance trade-off for numerical stability?

**Setup:**
- Array sizes: 1K, 10K, 100K, 1M, 10M elements
- Block size: 256 threads/block (optimal)
- Iterations: 1000 runs per configuration
- Metric: Median execution time

**Methods compared:**
1. **Naive:** 2 kernel launches
   - Kernel 1: Compute sum(exp(x)) via reduction
   - Kernel 2: Normalize (element-wise divide)

2. **Multi-pass:** 3 kernel launches
   - Kernel 1: Find max(x) via reduction
   - Kernel 2: Compute sum(exp(x - max)) via reduction
   - Kernel 3: Normalize with max adjustment

### Results

| Array Size | Naive (ms) | Multi-pass (ms) | Overhead |
|------------|------------|-----------------|----------|
| 1K | 0.035 | 0.060 | **+71%** |
| 10K | 0.039 | 0.060 | **+54%** |
| 100K | 0.047 | 0.083 | **+77%** |
| 1M | 0.190 | 0.367 | **+93%** |
| 10M | 0.260 | 0.477 | **+83%** |

### Key Findings

1. **Multi-pass is ~50-90% slower**
   - Extra kernel launch for max-finding
   - Additional memory passes
   - Kernel launch overhead dominates for small arrays

2. **Overhead is relatively constant**
   - ~70-90% across all sizes
   - Suggests fixed cost (kernel launches) dominates
   - Not bandwidth-limited yet (even at 10M elements)

3. **Performance breakdown (estimated for 1M elements):**
   ```
   Naive (0.190 ms):
     exp-sum reduction: ~0.100 ms
     normalize:         ~0.080 ms
     overhead:          ~0.010 ms

   Multi-pass (0.367 ms):
     max reduction:     ~0.090 ms  (extra!)
     exp-sum reduction: ~0.100 ms
     normalize:         ~0.080 ms
     overhead:          ~0.097 ms  (3 launches vs 2)
   ```

4. **Why multi-pass is slower:**
   - **3 kernel launches vs 2:** Each launch has ~0.01-0.02 ms overhead
   - **2 reduction passes:** Max + exp-sum (naive only does exp-sum)
   - **Memory traffic:** 2 full reads of input (max stage, exp-sum stage)

5. **The trade-off:**
   - **Naive:** Fast but **BROKEN** for large inputs
   - **Multi-pass:** Slower but **ALWAYS CORRECT**
   - **In production:** Correctness >> 2x performance
   - **Real workloads:** Transformer attention uses stable softmax exclusively

### Why Accept the Overhead?

1. **Correctness is non-negotiable**
   - NaN/Inf breaks training completely
   - Model divergence is catastrophic
   - Cannot predict when inputs will be large

2. **Relative cost is small**
   - Softmax is usually <1% of total model time
   - Attention compute dominates (matmuls)
   - 2x softmax overhead = 0.5% total slowdown

3. **Modern optimizations recover performance**
   - Fused kernels (future work)
   - Online algorithms (future work)
   - Warp-level primitives
   - Can match naive speed while staying stable!

---

## New CUDA Concepts Learned

### 1. Numerical Stability in GPU Computing

**Why it matters more on GPU:**
```cuda
// CPU: Often uses double (64-bit) by default
double sum = 0.0;
for (int i = 0; i < n; i++) {
    sum += exp(x[i]);  // Less likely to overflow
}

// GPU: Typically uses float (32-bit) for performance
float sum = 0.0f;
// exp(89) already overflows float32!
```

**The max subtraction trick:**
```cuda
// Find max first
float max_val = -INFINITY;
for (int i = 0; i < n; i++) {
    max_val = fmaxf(max_val, x[i]);
}

// Then compute exp with adjustment
for (int i = 0; i < n; i++) {
    result[i] = expf(x[i] - max_val);  // Safe!
}
```

### 2. Multi-stage Reduction Algorithms

**Pattern:** Reduce → Process → Normalize
```
Stage 1: Reduce to find statistic (max, sum, etc.)
Stage 2: Reduce with statistic (exp-sum using max)
Stage 3: Element-wise transform (normalize)
```

**When to use:**
- Softmax (max → exp-sum → normalize)
- Batch normalization (mean → variance → normalize)
- Layer normalization (mean → variance → normalize)

### 3. Reduction Operator Variants

**Max reduction vs Sum reduction:**
```cuda
// Sum reduction
sdata[tid] += sdata[tid + stride];

// Max reduction
sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);

// Min reduction
sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
```

**Key insight:** Same tree reduction pattern, different operator!

### 4. Handling Out-of-Bounds in Reductions

**Critical for correctness:**
```cuda
// Sum reduction: use identity element 0
sdata[tid] = (idx < n) ? input[idx] : 0.0f;

// Max reduction: use identity element -infinity
sdata[tid] = (idx < n) ? input[idx] : -INFINITY;

// Min reduction: use identity element +infinity
sdata[tid] = (idx < n) ? input[idx] : INFINITY;
```

### 5. Sequential Reduction Stages

**Bug we fixed:** Don't re-apply transformation on already-transformed values!
```cuda
// WRONG: Applies exp twice in subsequent stages
for stage in stages:
    expSumReductionKernel()  // Does exp every time!

// CORRECT: Only exp first stage
if (firstStage) {
    expSumReductionKernel()  // exp(x - max)
} else {
    sumReductionKernel()     // Just sum (already exp'd)
}
```

**Lesson:** Track transformation state across reduction stages!

---

## Future Work

### Planned Implementations (Skeletons Created)

**1. Fused Softmax:**
- Combine max and exp-sum in single kernel
- Use shared memory for block-level fusion
- Expected: 30-50% faster than multi-pass
- Challenges: Complex shared memory management, numerical precision

**2. Online Softmax:**
- Single-pass algorithm (streaming max + sum)
- Update running statistics as we go
- Expected: Fastest (single memory pass)
- Challenges: Most complex, precision-sensitive

**3. Warp-optimized variants:**
- Use `__shfl_down_sync` for final 32→1 reduction
- Eliminate `__syncthreads` barriers
- Expected: 8-10% speedup (like in reduction experiments)

### Benchmark Goals

**Target performance (1M elements):**
- Naive: 0.190 ms (baseline, but broken)
- Multi-pass: 0.367 ms (current, stable)
- Fused: ~0.250 ms (goal)
- Online: ~0.200 ms (goal)
- **Goal:** Match naive speed while staying stable!

### 2D Softmax (Batch Processing)

**Extension to batch × seq_len:**
```cuda
// Each block processes one sequence
__global__ void softmax2D(float *input, float *output, int batch, int seq_len) {
    int row = blockIdx.x;  // Which sequence
    // Softmax over input[row * seq_len : (row+1) * seq_len]
}
```

**Use case:** Transformer attention (batch_size × num_heads × seq_len)

---

## Key Takeaways

1. **Numerical stability is not optional**
   - Naive softmax breaks with real-world inputs
   - Max subtraction is the standard solution
   - Small overhead is acceptable for correctness

2. **Performance trade-offs:**
   - Multi-pass: 2x slower but always correct
   - Can be optimized with kernel fusion
   - Fused/online variants can match naive speed

3. **Design pattern:** Multi-stage reductions
   - Find statistic (max, mean, etc.)
   - Use statistic in second reduction
   - Apply final transformation
   - Common in normalization operations

4. **Implementation details matter:**
   - Don't re-apply transformations in reduction stages!
   - Use correct identity elements for reductions
   - Track state across multi-kernel algorithms

5. **Production reality:**
   - All ML frameworks use stable softmax
   - PyTorch/TensorFlow: Fused + online algorithms
   - Numerical correctness >> raw performance
   - But optimizations can recover performance!

---

*Last Updated: 2026-01-01*
