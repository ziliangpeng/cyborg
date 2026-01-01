#include "reduce_kernels.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <stdlib.h>

// Block-level reduction kernel using shared memory
// Each block reduces its elements to a single partial sum
__global__ void sumReductionKernel(const float *input, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (or 0 if out of bounds)
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    // Each iteration: half the threads sum pairs
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes this block's result
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// Option B: Fully GPU recursive reduction
// Keeps launching kernels until only 1 element remains
float vectorSum_GPU(const float *d_input, int n, int threadsPerBlock) {
    const float *d_current = d_input;
    int currentSize = n;
    bool allocated = false;

    // Keep reducing until we have 1 element
    while (currentSize > 1) {
        int numBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate output for this reduction stage
        float *d_output;
        cudaCheckError(cudaMalloc(&d_output, numBlocks * sizeof(float)));

        // Launch reduction kernel
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        sumReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
            d_current, d_output, currentSize);
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

// Option D: GPU with configurable CPU threshold
// Reduces on GPU until size <= threshold, then finishes on CPU
float vectorSum_Threshold(const float *d_input, int n, int threadsPerBlock, int cpuThreshold) {
    const float *d_current = d_input;
    int currentSize = n;
    bool allocated = false;

    // Keep reducing on GPU while size > threshold
    while (currentSize > cpuThreshold) {
        int numBlocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate output for this reduction stage
        float *d_output;
        cudaCheckError(cudaMalloc(&d_output, numBlocks * sizeof(float)));

        // Launch reduction kernel
        size_t sharedMemSize = threadsPerBlock * sizeof(float);
        sumReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
            d_current, d_output, currentSize);
        cudaCheckError(cudaGetLastError());

        // Free previous buffer if we allocated it
        if (allocated) {
            cudaCheckError(cudaFree((void*)d_current));
        }

        // Move to next stage
        d_current = d_output;
        currentSize = numBlocks;
        allocated = true;
    }

    // Transfer remaining elements to CPU and finish reduction
    float *h_partial = (float*)malloc(currentSize * sizeof(float));
    cudaCheckError(cudaMemcpy(h_partial, d_current, currentSize * sizeof(float),
                               cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    float result = 0.0f;
    for (int i = 0; i < currentSize; i++) {
        result += h_partial[i];
    }

    // Cleanup
    free(h_partial);
    if (allocated) {
        cudaCheckError(cudaFree((void*)d_current));
    }

    return result;
}
