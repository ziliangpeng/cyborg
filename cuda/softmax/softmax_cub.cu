#include "softmax_cub.h"
#include "cuda_utils.h"
#include "elementwise_kernels.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>  // CUB library
#include <stdlib.h>
#include <math.h>

// ============================================================================
// CUB-BASED SOFTMAX IMPLEMENTATION (3-kernel, CUB-optimized)
// ============================================================================
//
// WHAT IS CUB?
// ------------
// CUB (CUDA Unbound) is NVIDIA's library of highly optimized CUDA primitives.
// It's now part of CUDA Core Compute Libraries (CCCL) and included in CUDA Toolkit.
//
// Key features:
// - BlockReduce: Optimized block-level reductions (sum, max, min, etc.)
// - WarpReduce: Warp-level reductions using shuffle instructions
// - Automatic algorithm selection based on block size
// - Template-based with compile-time optimization
// - Zero runtime overhead compared to hand-written code
//
// WHY USE CUB FOR SOFTMAX?
// ------------------------
// 1. Less code: CUB manages shared memory allocation and synchronization
// 2. Better performance: NVIDIA engineers optimized for all GPU architectures
// 3. More maintainable: Changes to block size don't require rewriting reduction logic
// 4. Proven correctness: Extensively tested across GPU generations
//
// COMPARISON TO HAND-WRITTEN REDUCTIONS:
// --------------------------------------
// Our fused3 implementation:
//   - Manual tree reduction with explicit shared memory management
//   - ~30 lines of reduction code per kernel
//   - Must handle synchronization, bank conflicts, boundary conditions
//
// CUB implementation:
//   - Single line: BlockReduce(temp_storage).Reduce(value, cub::Max())
//   - CUB handles all complexity internally
//   - Automatically uses warp shuffles when beneficial
//
// ALGORITHM OVERVIEW:
// ------------------
// Same 3-kernel architecture as fused3, but with CUB-optimized reductions:
//
// Kernel 1 - softmaxCub_BlockStats:
//   Each block processes 256 elements (or threadsPerBlock)
//   Phase 1: Find block max using CUB BlockReduce::Reduce(val, cub::Max())
//   Phase 2: Compute sum(exp(x - block_max)) using CUB BlockReduce::Sum()
//   Output: block_maxes[blockIdx], block_sums[blockIdx]
//
// Kernel 2 - softmaxCub_GlobalReduce:
//   Single block processes all block statistics
//   Phase 1: Find global max using CUB BlockReduce with grid-stride loop
//   Phase 2: Compute adjusted global sum with exponential correction
//   Output: global_max[0], global_sum[0]
//
// Kernel 3 - softmaxNormalizeKernel (reused from elementwise_kernels.h):
//   Each thread normalizes one element: output[i] = exp(x[i] - max) / sum
//
// CUB BLOCKREDUCE PATTERN:
// ------------------------
// typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
// __shared__ typename BlockReduce::TempStorage temp_storage;
//
// float result = BlockReduce(temp_storage).Reduce(value, cub::Max());
// // or
// float result = BlockReduce(temp_storage).Sum(value);
//
// Key points:
// - TempStorage is CUB's way of managing shared memory
// - BlockReduce constructor takes temp_storage reference
// - Reduce() or Sum() performs the actual reduction
// - Result is valid only in thread 0 (unless using AllReduce pattern)
//
// PERFORMANCE CHARACTERISTICS:
// ----------------------------
// Expected: Similar to fused3 (possibly 1-5% faster due to CUB optimizations)
// - CUB uses warp shuffles when block size is power of 2
// - Handles bank conflicts automatically
// - Template specialization eliminates runtime overhead
//
// CUB vs hand-written (from NVIDIA benchmarks):
// - Competitive or better performance
// - 10-50% less code
// - Better portability across GPU architectures
//
// ============================================================================

// Kernel 1: Compute block-level statistics using CUB
template<int BLOCK_SIZE>
__global__ void softmaxCub_BlockStats(
    const float *input,
    float *block_maxes,
    float *block_sums,
    int n
) {
    // CUB BlockReduce type definition
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;

    // Allocate shared memory for CUB's temporary storage
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: Find block maximum using CUB
    float thread_max = (idx < n) ? input[idx] : -INFINITY;
    float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    // Only thread 0 has the valid result - broadcast to shared memory
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = block_max;
        block_maxes[blockIdx.x] = block_max;
    }
    __syncthreads();
    block_max = shared_max;  // All threads now have the max

    // Phase 2: Compute sum(exp(x - block_max)) using CUB
    // Note: We reuse temp_storage here - legal after __syncthreads()
    float thread_exp_sum = (idx < n) ? expf(input[idx] - block_max) : 0.0f;
    float block_sum = BlockReduce(temp_storage).Sum(thread_exp_sum);

    // Thread 0 writes the sum
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = block_sum;
    }
}

// Kernel 2: Global reduce using CUB (single block processes all block stats)
template<int BLOCK_SIZE>
__global__ void softmaxCub_GlobalReduce(
    const float *block_maxes,
    const float *block_sums,
    float *global_max,
    float *global_sum,
    int numBlocks
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;

    // Phase 1: Find global max using grid-stride loop + CUB reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < numBlocks; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, block_maxes[i]);
    }

    float global_max_val = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    __shared__ float shared_global_max;
    if (tid == 0) {
        shared_global_max = global_max_val;
        global_max[0] = global_max_val;
    }
    __syncthreads();
    global_max_val = shared_global_max;

    // Phase 2: Compute adjusted global sum using grid-stride loop + CUB reduction
    // Key formula: adjusted_sum = Σ(block_sum[i] * exp(block_max[i] - global_max))
    float thread_sum = 0.0f;
    for (int i = tid; i < numBlocks; i += BLOCK_SIZE) {
        thread_sum += block_sums[i] * expf(block_maxes[i] - global_max_val);
    }

    // Reuse temp_storage after __syncthreads()
    float global_sum_val = BlockReduce(temp_storage).Sum(thread_sum);

    if (tid == 0) {
        global_sum[0] = global_sum_val;
    }
}

// Host function: CUB-based softmax
float softmax_Cub(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    // Calculate grid dimensions
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate intermediate buffers
    float *d_block_maxes, *d_block_sums;
    cudaCheckError(cudaMalloc(&d_block_maxes, numBlocks * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));

    float *d_global_max, *d_global_sum;
    cudaCheckError(cudaMalloc(&d_global_max, sizeof(float)));
    cudaCheckError(cudaMalloc(&d_global_sum, sizeof(float)));

    // Launch Kernel 1: Block statistics with CUB
    // Note: We use template specialization for common block sizes
    if (threadsPerBlock == 256) {
        softmaxCub_BlockStats<256><<<numBlocks, 256>>>(
            d_input, d_block_maxes, d_block_sums, n);
    } else if (threadsPerBlock == 128) {
        softmaxCub_BlockStats<128><<<numBlocks, 128>>>(
            d_input, d_block_maxes, d_block_sums, n);
    } else if (threadsPerBlock == 512) {
        softmaxCub_BlockStats<512><<<numBlocks, 512>>>(
            d_input, d_block_maxes, d_block_sums, n);
    } else {
        // Fallback for other block sizes (will be slower due to dynamic template instantiation)
        printf("Warning: Block size %d not optimized. Consider using 128, 256, or 512.\n", threadsPerBlock);
        softmaxCub_BlockStats<256><<<numBlocks, threadsPerBlock>>>(
            d_input, d_block_maxes, d_block_sums, n);
    }
    cudaCheckError(cudaGetLastError());

    // Launch Kernel 2: Global reduce with CUB (single block)
    if (threadsPerBlock == 256) {
        softmaxCub_GlobalReduce<256><<<1, 256>>>(
            d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    } else if (threadsPerBlock == 128) {
        softmaxCub_GlobalReduce<128><<<1, 128>>>(
            d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    } else if (threadsPerBlock == 512) {
        softmaxCub_GlobalReduce<512><<<1, 512>>>(
            d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    } else {
        softmaxCub_GlobalReduce<256><<<1, threadsPerBlock>>>(
            d_block_maxes, d_block_sums, d_global_max, d_global_sum, numBlocks);
    }
    cudaCheckError(cudaGetLastError());

    // Copy global statistics to host for Kernel 3
    float h_global_max, h_global_sum;
    cudaCheckError(cudaMemcpy(&h_global_max, d_global_max, sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(&h_global_sum, d_global_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // Launch Kernel 3: Normalize (reuse existing kernel)
    softmaxNormalizeKernel<<<numBlocks, threadsPerBlock>>>(
        d_input, h_global_max, h_global_sum, d_output, n);
    cudaCheckError(cudaGetLastError());

    // Cleanup
    cudaCheckError(cudaFree(d_block_maxes));
    cudaCheckError(cudaFree(d_block_sums));
    cudaCheckError(cudaFree(d_global_max));
    cudaCheckError(cudaFree(d_global_sum));

    return 0.0f;  // Timing handled by caller
}
