#include "softmax_naive.h"
#include "cuda_utils.h"
#include "reduce_kernels.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// NAIVE SOFTMAX IMPLEMENTATION (Numerically Unstable - For Educational Demo)
// ============================================================================
//
// ALGORITHM OVERVIEW:
// ------------------
// Computes softmax as: output[i] = exp(x[i]) / sum(exp(x[j]))
//
// Two-stage process:
//   Stage 1: Compute sum(exp(x)) via recursive reduction
//   Stage 2: Normalize by dividing exp(x[i]) / sum
//
// IMPLEMENTATION DETAILS:
// -----------------------
// Stage 1 - Sum Reduction (2-4 kernel launches depending on size):
//   - Launch 1: expSumReductionKernel
//       * Computes exp(x[i]) for each element
//       * Reduces within each block using warp-optimized tree reduction
//       * Outputs partial sums: n elements → numBlocks partial sums
//   - Launch 2+: sumReductionKernel_Warp (recursive)
//       * Sums the partial results from previous stage
//       * Does NOT apply exp() again (values already exponentiated)
//       * Continues until only 1 final sum remains
//
// Stage 2 - Normalization (1 kernel launch):
//   - naiveNormalizeKernel
//       * Computes output[i] = exp(x[i]) / sum
//       * Each thread handles one element independently
//       * Note: Recomputes exp(x[i]) - inefficient but simple
//
// WARP OPTIMIZATIONS:
// -------------------
// Both kernels use warp shuffle primitives for the final 32→1 reduction:
//   - Shared memory reduction: blockDim → 32 elements
//   - Warp shuffle reduction: 32 → 1 element (no __syncthreads needed)
//   - Eliminates 5 __syncthreads barriers, ~8% speedup
//
// WHY IT'S NUMERICALLY UNSTABLE:
// -------------------------------
// Problem: Computes exp(x) directly without max subtraction
//   - If x[i] = 89: exp(89) ≈ 4.5e38 (near float32 max)
//   - If x[i] = 90: exp(90) = Inf (OVERFLOW!)
//   - Result: NaN/Inf in output, completely broken
//
// Example failure with input [88, 89, 90]:
//   exp(88) = 1.7e38   ✓
//   exp(89) = 4.5e38   ✓ (barely fits in float32)
//   exp(90) = Inf      ✗ OVERFLOW
//   Sum = Inf
//   Output = [Inf/Inf, Inf/Inf, Inf/Inf] = [NaN, NaN, NaN]
//
// KERNEL LAUNCH COUNT:
// --------------------
// For 1M elements with 256 threads/block:
//   Launch 1: expSumReductionKernel (1M → 3,907 blocks)
//   Launch 2: sumReductionKernel_Warp (3,907 → 16 blocks)
//   Launch 3: sumReductionKernel_Warp (16 → 1 block)
//   Launch 4: naiveNormalizeKernel (1M elements)
//   Total: 4 kernel launches
//
// PERFORMANCE:
// ------------
// - Fast: ~0.187ms for 1M elements (median)
// - Fastest method for small-medium sizes (1K-1M)
// - But produces WRONG RESULTS (NaN/Inf) for inputs > 88
// - Only works for very small input ranges (< 10)
//
// Use stable methods (multi-pass or fused) for production code!
//
// ============================================================================

// Note: sumReductionKernel_Warp is imported from reduce_kernels.h

// Kernel: Compute exp(x) and reduce to sum using warp-optimized tree reduction
__global__ void expSumReductionKernel(const float *input, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input and compute exp (no max subtraction - UNSTABLE!)
    sdata[tid] = (idx < n) ? expf(input[idx]) : 0.0f;
    __syncthreads();

    // Part 1: Shared memory reduction (blockDim → 64)
    // Stop before entering warp-level range
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Explicit stride=32 reduction (64 → 32)
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
    }
    __syncthreads();

    // Part 2: Warp-level reduction for final 32 → 1 (no __syncthreads needed!)
    if (tid < 32) {
        float val = sdata[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);  // 32 → 16
        val += __shfl_down_sync(0xffffffff, val, 8);   // 16 → 8
        val += __shfl_down_sync(0xffffffff, val, 4);   // 8 → 4
        val += __shfl_down_sync(0xffffffff, val, 2);   // 4 → 2
        val += __shfl_down_sync(0xffffffff, val, 1);   // 2 → 1

        // Thread 0 writes this block's partial sum
        if (tid == 0) {
            partialSums[blockIdx.x] = val;
        }
    }
}

// Kernel: Normalize by dividing exp(x) by sum (naive - no max subtraction)
__global__ void naiveNormalizeKernel(const float *input, float sum_exp, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = expf(input[idx]) / sum_exp;
    }
}

// Host function: Naive softmax (demonstrates overflow)
float softmax_Naive(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    const float *d_current = d_input;
    int currentSize = n;
    bool allocated = false;
    bool firstStage = true;

    // Stage 1: Compute sum(exp(x)) using reduction
    while (currentSize > 1) {
        int numBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate output for this reduction stage
        float *d_output_stage;
        cudaCheckError(cudaMalloc(&d_output_stage, numBlocks * sizeof(float)));

        // Launch appropriate kernel
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        if (firstStage) {
            // First stage: compute exp(x) from original input
            expSumReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output_stage, currentSize);
            firstStage = false;
        } else {
            // Subsequent stages: just sum partial results (already exp'd)
            // Use warp-optimized kernel for ~8% speedup
            sumReductionKernel_Warp<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output_stage, currentSize);
        }
        cudaCheckError(cudaGetLastError());

        // Free previous buffer if we allocated it
        if (allocated) {
            cudaCheckError(cudaFree((void*)d_current));
        }

        // Move to next stage
        d_current = d_output_stage;
        currentSize = numBlocks;
        allocated = true;
    }

    // Copy final sum to host
    float sum_exp;
    cudaCheckError(cudaMemcpy(&sum_exp, d_current, sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup reduction buffer
    if (allocated) {
        cudaCheckError(cudaFree((void*)d_current));
    }

    // Stage 2: Normalize (divide by sum)
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    naiveNormalizeKernel<<<numBlocks, threadsPerBlock>>>(
        d_input, sum_exp, d_output, n);
    cudaCheckError(cudaGetLastError());
    cudaDeviceSynchronize();

    return 0.0f;  // Timing handled by caller
}
