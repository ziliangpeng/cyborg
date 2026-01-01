#include "softmax_kernels.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// NAIVE SOFTMAX IMPLEMENTATION (Numerically Unstable - For Educational Demo)
// ============================================================================

// Kernel: Simple sum reduction (for already-computed values)
__global__ void sumReductionKernel(const float *input, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input (no transformation)
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction to sum values
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

// ============================================================================
// MULTI-PASS SOFTMAX IMPLEMENTATION (Numerically Stable)
// ============================================================================

// Kernel: Max reduction using tree reduction pattern
__global__ void maxReductionKernel(const float *input, float *partialMaxs, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (or -infinity if out of bounds)
    sdata[tid] = (idx < n) ? input[idx] : -INFINITY;
    __syncthreads();

    // Tree reduction with max operator
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes this block's partial max
    if (tid == 0) {
        partialMaxs[blockIdx.x] = sdata[0];
    }
}

// Kernel: Compute exp(x - max) and reduce to sum
__global__ void expSumReductionKernel_Stable(const float *input, float max_val, float *partialSums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input and compute exp(x - max) for numerical stability
    sdata[tid] = (idx < n) ? expf(input[idx] - max_val) : 0.0f;
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

// Kernel: Normalize with stable formula: exp(x - max) / sum
__global__ void softmaxNormalizeKernel(const float *input, float max_val, float sum_exp, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}

// Helper: GPU reduction to find max value
float vectorMax_GPU(const float *d_input, int n, int threadsPerBlock) {
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
        maxReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
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
            sumReductionKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
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

// Host function: Multi-pass stable softmax
float softmax_MultiPass(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    // Stage 1: Find max(x)
    float max_val = vectorMax_GPU(d_input, n, threadsPerBlock);

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

// ============================================================================
// FUSED SOFTMAX IMPLEMENTATION (SKELETON - To Be Implemented)
// ============================================================================

/*
 * TODO: Implement fused softmax kernel
 *
 * Algorithm:
 * 1. Each block processes chunk of data in shared memory
 * 2. Block-level reduction to find block_max
 * 3. Block-level reduction to compute block_sum = sum(exp(x - block_max))
 * 4. Write partial results (block_max, block_sum) to global memory
 * 5. Reduce block results to get global_max and global_sum
 * 6. Final normalization pass with global statistics
 *
 * Benefits:
 * - Fewer kernel launches (1-2 vs 3)
 * - Less memory traffic (fewer passes over data)
 * - Block-level fusion reduces global memory accesses
 *
 * Challenges:
 * - More complex shared memory management
 * - Need to adjust block_sum when merging: block_sum * exp(block_max - global_max)
 * - Careful with numerical precision
 */

float softmax_Fused(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    printf("ERROR: Fused softmax not implemented yet\n");
    printf("This is a skeleton for future implementation\n");
    return 0.0f;
}

// ============================================================================
// ONLINE SOFTMAX IMPLEMENTATION (SKELETON - To Be Implemented)
// ============================================================================

/*
 * TODO: Implement online softmax kernel
 *
 * Algorithm (streaming computation):
 * Initialize: running_max = -infinity, running_sum = 0
 *
 * For each element x[i]:
 *   old_max = running_max
 *   running_max = max(running_max, x[i])
 *   running_sum = running_sum * exp(old_max - running_max) + exp(x[i] - running_max)
 *
 * At end: running_sum is correct sum for running_max
 * Normalize: output[i] = exp(x[i] - running_max) / running_sum
 *
 * Benefits:
 * - Single pass over data (most memory efficient)
 * - Numerically stable (maintains max and adjusted sum)
 * - Elegant algorithm (teaches online statistics)
 *
 * Challenges:
 * - Most complex to implement
 * - Block-level online algorithm + merge step
 * - Floating point precision sensitive
 * - Need careful testing of numerical stability
 */

float softmax_Online(const float *d_input, float *d_output, int n, int threadsPerBlock) {
    printf("ERROR: Online softmax not implemented yet\n");
    printf("This is a skeleton for future implementation\n");
    return 0.0f;
}
