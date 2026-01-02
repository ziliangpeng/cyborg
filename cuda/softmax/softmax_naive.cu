#include "softmax_naive.h"
#include "cuda_utils.h"
#include "reduce_kernels.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// NAIVE SOFTMAX IMPLEMENTATION (Numerically Unstable - For Educational Demo)
// ============================================================================

// Note: sumReductionKernel is now imported from reduce_kernels.h (no duplication)

// Kernel: Compute exp(x) and reduce to sum using tree reduction
__global__ void expSumReductionKernel(const float *input, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input and compute exp (no max subtraction - UNSTABLE!)
    sdata[tid] = (idx < n) ? expf(input[idx]) : 0.0f;
    __syncthreads();

    // Tree reduction to sum exp values
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes this block's partial sum
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
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
            sumReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
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
