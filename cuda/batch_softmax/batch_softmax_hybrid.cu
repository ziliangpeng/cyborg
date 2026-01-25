#include "batch_softmax_hybrid.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// HYBRID BATCH SOFTMAX: Adaptive Kernel Selection
// ============================================================================
//
// This implementation provides a single interface that automatically selects
// the optimal kernel based on the dimension size at construction time.
//
// KERNEL SELECTION STRATEGY:
// --------------------------
// dim <= 64:    Warp kernel (32 threads)
//   - Each thread processes <=2 elements
//   - Zero shared memory, zero __syncthreads()
//   - Minimal launch overhead
//   - Best: Low latency for small dims
//
// dim <= 1024:  Multi-warp scalar kernel (256 threads)
//   - 8 warps = more parallelism
//   - Hybrid reduction (warp shuffles + minimal shared mem)
//   - Good balance of throughput and overhead
//   - Best: Medium dims where ILP helps
//
// dim > 1024:   Multi-warp vectorized kernel (256 threads + float4)
//   - Vectorized loads for 4x memory bandwidth
//   - Same hybrid reduction as scalar
//   - Requires dim % 4 == 0 (falls back to scalar otherwise)
//   - Best: Large dims where memory bandwidth dominates
//
// THRESHOLD RATIONALE:
// --------------------
// - 64: At 32 threads, each thread handles dim/32 elements. At dim=64,
//   that's only 2 elements per thread - perfect for warp kernel.
//   Beyond 64, more threads help with instruction overlap.
//
// - 1024: At this point, memory bandwidth becomes the bottleneck.
//   Vectorized loads (float4) provide ~2-3x speedup for memory-bound cases.
//   Below 1024, the overhead of vectorization may not be worth it.
//
// These thresholds are tuned for modern GPUs (Volta, Ampere, Hopper).
// Actual optimal crossover points vary by:
// - GPU memory bandwidth
// - Cache size and hit rate
// - Number of SMs
//
// ============================================================================

#define FULL_MASK 0xffffffff
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)  // 256 threads

// Kernel 0: Warp kernel (32 threads per row)
__global__ void hybrid_warp_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;
    int tid = threadIdx.x;

    // Phase 1: Find max
    float thread_max = -INFINITY;
    for (int i = tid; i < dim; i += 32) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(FULL_MASK, thread_max, offset);
        thread_max = fmaxf(thread_max, other);
    }
    float row_max = __shfl_sync(FULL_MASK, thread_max, 0);

    // Phase 2: Compute sum
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += 32) {
        thread_sum += expf(row_input[i] - row_max);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(FULL_MASK, thread_sum, offset);
    }
    float row_sum = __shfl_sync(FULL_MASK, thread_sum, 0);

    // Phase 3: Normalize
    for (int i = tid; i < dim; i += 32) {
        row_output[i] = expf(row_input[i] - row_max) / row_sum;
    }
}

// Kernel 1: Multi-warp scalar kernel (256 threads per row)
__global__ void hybrid_multi_warp_scalar_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    __shared__ float warp_maxes[WARPS_PER_BLOCK];
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    // Phase 1: Find max
    float thread_max = -INFINITY;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(FULL_MASK, thread_max, offset));
    }

    if (lane_id == 0) warp_maxes[warp_id] = thread_max;
    __syncthreads();

    __shared__ float row_max_shared;
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? warp_maxes[lane_id] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
        }
        if (lane_id == 0) row_max_shared = val;
    }
    __syncthreads();
    float row_max = row_max_shared;

    // Phase 2: Compute sum
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        thread_sum += expf(row_input[i] - row_max);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(FULL_MASK, thread_sum, offset);
    }

    if (lane_id == 0) warp_sums[warp_id] = thread_sum;
    __syncthreads();

    __shared__ float row_sum_shared;
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        if (lane_id == 0) row_sum_shared = val;
    }
    __syncthreads();
    float row_sum = row_sum_shared;

    // Phase 3: Normalize
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        row_output[i] = expf(row_input[i] - row_max) / row_sum;
    }
}

// Kernel 2: Multi-warp vectorized kernel (256 threads + float4)
__global__ void hybrid_multi_warp_vec4_kernel(
    const float *input,
    float *output,
    int batch_size,
    int dim
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_input = input + row * dim;
    float *row_output = output + row * dim;
    const float4 *row_input4 = reinterpret_cast<const float4*>(row_input);
    float4 *row_output4 = reinterpret_cast<float4*>(row_output);
    int dim4 = dim / 4;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    __shared__ float warp_maxes[WARPS_PER_BLOCK];
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    // Phase 1: Find max with vectorized loads
    float thread_max = -INFINITY;
    for (int i = tid; i < dim4; i += BLOCK_SIZE) {
        float4 vals = row_input4[i];
        float local_max = fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w));
        thread_max = fmaxf(thread_max, local_max);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(FULL_MASK, thread_max, offset));
    }

    if (lane_id == 0) warp_maxes[warp_id] = thread_max;
    __syncthreads();

    __shared__ float row_max_shared;
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? warp_maxes[lane_id] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
        }
        if (lane_id == 0) row_max_shared = val;
    }
    __syncthreads();
    float row_max = row_max_shared;

    // Phase 2: Compute sum with vectorized loads
    float thread_sum = 0.0f;
    for (int i = tid; i < dim4; i += BLOCK_SIZE) {
        float4 vals = row_input4[i];
        thread_sum += expf(vals.x - row_max) + expf(vals.y - row_max) +
                      expf(vals.z - row_max) + expf(vals.w - row_max);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(FULL_MASK, thread_sum, offset);
    }

    if (lane_id == 0) warp_sums[warp_id] = thread_sum;
    __syncthreads();

    __shared__ float row_sum_shared;
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        if (lane_id == 0) row_sum_shared = val;
    }
    __syncthreads();
    float row_sum = row_sum_shared;

    // Phase 3: Normalize with vectorized stores
    for (int i = tid; i < dim4; i += BLOCK_SIZE) {
        float4 vals = row_input4[i];
        float4 out;
        out.x = expf(vals.x - row_max) / row_sum;
        out.y = expf(vals.y - row_max) / row_sum;
        out.z = expf(vals.z - row_max) / row_sum;
        out.w = expf(vals.w - row_max) / row_sum;
        row_output4[i] = out;
    }
}

// ============================================================================
// CLASS-BASED IMPLEMENTATION
// ============================================================================

HybridBatchSoftmax::HybridBatchSoftmax(int batch_size, int dim, int threadsPerBlock)
    : batch_size(batch_size), dim(dim) {
    (void)threadsPerBlock;  // Ignored - we select automatically

    // Select kernel based on dimension size
    if (dim <= 64) {
        selected_kernel = 0;  // Warp kernel
    } else if (dim <= 1024) {
        selected_kernel = 1;  // Multi-warp scalar
    } else {
        // Large dims: use vectorized if possible
        if (dim % 4 == 0) {
            selected_kernel = 2;  // Multi-warp vectorized
        } else {
            selected_kernel = 1;  // Fall back to scalar
        }
    }
}

void HybridBatchSoftmax::execute(const float *d_input, float *d_output) {
    switch (selected_kernel) {
        case 0:
            hybrid_warp_kernel<<<batch_size, 32>>>(
                d_input, d_output, batch_size, dim);
            break;
        case 1:
            hybrid_multi_warp_scalar_kernel<<<batch_size, BLOCK_SIZE>>>(
                d_input, d_output, batch_size, dim);
            break;
        case 2:
            hybrid_multi_warp_vec4_kernel<<<batch_size, BLOCK_SIZE>>>(
                d_input, d_output, batch_size, dim);
            break;
    }
    cudaCheckError(cudaGetLastError());
}
