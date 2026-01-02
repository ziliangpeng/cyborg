#include "softmax_multipass.h"
#include "cuda_utils.h"
#include "reduce_kernels.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// MULTI-PASS SOFTMAX IMPLEMENTATION (Numerically Stable)
// ============================================================================

// Note: maxReductionKernel and vectorMax_GPU are now imported from reduce_kernels.h

// Kernel: Compute exp(x - max) and reduce to sum (warp-optimized)
__global__ void expSumReductionKernel_Stable(const float *input, float max_val, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input and compute exp(x - max) for numerical stability
    sdata[tid] = (idx < n) ? expf(input[idx] - max_val) : 0.0f;
    __syncthreads();

    // Part 1: Shared memory reduction (blockDim → 64)
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
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);

        if (tid == 0) {
            partialSums[blockIdx.x] = val;
        }
    }
}

// Kernel: Normalize with stable formula: exp(x - max) / sum
__global__ void softmaxNormalizeKernel(const float *input, float max_val, float sum_exp, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}

// Note: vectorMax_GPU is now provided by reduce_kernels.h

// Helper: GPU reduction to compute sum(exp(x - max))
float vectorExpSum_GPU(const float *d_input, float max_val, int n, int threadsPerBlock) {
    const float *d_current = d_input;
    int currentSize = n;
    bool allocated = false;
    bool firstStage = true;

    // Keep reducing until we have 1 element
    while (currentSize > 1) {
        int numBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate output for this reduction stage
        float *d_output;
        cudaCheckError(cudaMalloc(&d_output, numBlocks * sizeof(float)));

        // Launch appropriate kernel
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        if (firstStage) {
            // First stage: compute exp(x - max) from original input
            expSumReductionKernel_Stable<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, max_val, d_output, currentSize);
            firstStage = false;
        } else {
            // Subsequent stages: just sum partial results (already exp'd, don't exp again!)
            // Use warp-optimized kernel for ~8% speedup
            sumReductionKernel_Warp<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
                d_current, d_output, currentSize);
        }
        cudaCheckError(cudaGetLastError());

        // Free previous temp buffer (if we allocated it)
        if (allocated) {
            cudaCheckError(cudaFree((void*)d_current));
        }

        // Move to next stage
        d_current = d_output;
        currentSize = numBlocks;
        allocated = true;
    }

    // Copy final single element to host
    float result;
    cudaCheckError(cudaMemcpy(&result, d_current, sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup final buffer
    if (allocated) {
        cudaCheckError(cudaFree((void*)d_current));
    }

    return result;
}

// Host function: Multi-pass stable softmax (warp-optimized)
float softmax_MultiPass(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    // Stage 1: Find max(x) using warp-optimized kernel
    float max_val = vectorMax_GPU_Warp(d_input, n, threadsPerBlock);

    // Stage 2: Compute sum(exp(x - max))
    float sum_exp = vectorExpSum_GPU(d_input, max_val, n, threadsPerBlock);

    // Debug output (commented out after fixing overflow bug)
    // printf("[DEBUG] n=%d, max_val=%f, sum_exp=%f\n", n, max_val, sum_exp);

    // Stage 3: Normalize: output[i] = exp(x[i] - max) / sum
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    softmaxNormalizeKernel<<<numBlocks, threadsPerBlock>>>(
        d_input, max_val, sum_exp, d_output, n);
    cudaCheckError(cudaGetLastError());
    cudaDeviceSynchronize();

    return 0.0f;  // Timing handled by caller
}
