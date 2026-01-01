#include "vector_kernels.h"

// CUDA kernel: runs on the GPU
// Each thread computes one element of the result
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate global thread ID
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for vector multiplication
__global__ void vectorMul(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel for fused multiply-add (FMA)
// Computes d[i] = a[i] * b[i] + c[i] in a single kernel
__global__ void vectorFMA(const float *a, const float *b, const float *c, float *d, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        d[idx] = a[idx] * b[idx] + c[idx];
    }
}
