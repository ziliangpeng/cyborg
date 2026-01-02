#include "softmax_fused.h"
#include "cuda_utils.h"
#include "reduce_kernels.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// FUSED SOFTMAX IMPLEMENTATION (3-kernel optimized)
// ============================================================================

// Kernel 1: Compute block-level statistics (max and exp-sum)
__global__ void softmaxFused_BlockStats(
    const float *input,
    float *block_maxes,
    float *block_sums,
    int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: Find block maximum using tree reduction
    // Load input data (use -INFINITY for out-of-bounds)
    sdata[tid] = (idx < n) ? input[idx] : -INFINITY;
    __syncthreads();

    // Tree reduction to find max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // All threads read the block max
    float block_max = sdata[0];
    __syncthreads();

    // Phase 2: Compute sum(exp(x - block_max)) using tree reduction
    // Reuse shared memory - load and transform data
    sdata[tid] = (idx < n) ? expf(input[idx] - block_max) : 0.0f;
    __syncthreads();

    // Tree reduction to sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Phase 3: Write block statistics
    if (tid == 0) {
        block_maxes[blockIdx.x] = block_max;
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Reduce block statistics to global max and adjusted global sum
__global__ void softmaxFused_GlobalReduce(
    const float *block_maxes,
    const float *block_sums,
    float *global_max,
    float *global_sum,
    int numBlocks
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Phase 1: Find global maximum
    // Each thread finds max over multiple blocks using grid-stride loop
    float thread_max = -INFINITY;
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        thread_max = fmaxf(thread_max, block_maxes[i]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    // Tree reduction to find max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // All threads read the global max
    float max_val = sdata[0];
    __syncthreads();

    // Write global max (done once)
    if (tid == 0) {
        global_max[0] = max_val;
    }

    // Phase 2: Compute adjusted global sum
    // Critical formula: adjusted_sum = block_sum * exp(block_max - global_max)
    // Each thread sums over multiple blocks using grid-stride loop
    float thread_sum = 0.0f;
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        thread_sum += block_sums[i] * expf(block_maxes[i] - max_val);
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Tree reduction to sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write global sum
    if (tid == 0) {
        global_sum[0] = sdata[0];
    }
}

// Kernel 3: Final normalization using global statistics
__global__ void softmaxFused_Normalize(
    const float *input,
    const float *global_max,
    const float *global_sum,
    float *output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float max_val = global_max[0];
        float sum_val = global_sum[0];
        output[idx] = expf(input[idx] - max_val) / sum_val;
    }
}

// Host function: 3-kernel fused softmax
float softmax_Fused(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    // Calculate grid dimensions
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate intermediate buffers for block statistics
    float *d_block_maxes, *d_block_sums;
    cudaCheckError(cudaMalloc(&d_block_maxes, numBlocks * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));

    // Allocate buffers for global statistics
    float *d_global_max, *d_global_sum;
    cudaCheckError(cudaMalloc(&d_global_max, sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_sum, sizeof(float)));

    // Calculate shared memory size
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch Kernel 1: Compute block statistics
    softmaxFused_BlockStats<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_input, d_block_maxes, d_block_sums, n);
    cudaCheckError(cudaGetLastError());

    // Launch Kernel 2: Reduce to global statistics (single block)
    softmaxFused_GlobalReduce<<<1, threadsPerBlock, sharedMemSize>>>(
        d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    cudaCheckError(cudaGetLastError());

    // Launch Kernel 3: Final normalization
    softmaxFused_Normalize<<<numBlocks, threadsPerBlock>>>(
        d_input, d_global_max, d_global_sum, d_output, n);
    cudaCheckError(cudaGetLastError());

    // Synchronize before cleanup
    cudaDeviceSynchronize();

    // Cleanup intermediate buffers
    cudaCheckError(cudaFree(d_block_maxes));
    cudaCheckError(cudaFree(d_block_sums));
    cudaCheckError(cudaFree(d_global_max));
    cudaCheckError(cudaFree(d_global_sum));

    return 0.0f;  // Timing handled by caller
}
