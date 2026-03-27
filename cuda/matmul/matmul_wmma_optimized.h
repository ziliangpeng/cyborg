#ifndef MATMUL_WMMA_OPTIMIZED_H
#define MATMUL_WMMA_OPTIMIZED_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>

// Optimized WMMA Tensor Core matmul kernel
//
// OPTIMIZATIONS:
// 1. Large block tiles (128x256) with shared memory staging
// 2. Double buffering for memory latency hiding
// 3. Asynchronous memory copies (cp.async) for Hopper
// 4. Multiple WMMA tiles per warp (2x4 = 8 tiles)
// 5. Register-level accumulator tiling
// 6. Shared memory swizzling to avoid bank conflicts
//
// PERFORMANCE TARGET: >30% MFU on H100 (vs 5% for basic WMMA)
//
// REQUIREMENTS:
// - SM80+ (Ampere or newer for cp.async, BF16)
// - N must be multiple of 128

class MatmulWmmaOptimized : public MatmulKernel {
private:
    int N;
    half *d_A_fp16;
    half *d_B_fp16;

public:
    MatmulWmmaOptimized(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptimized() override;
};

#endif
