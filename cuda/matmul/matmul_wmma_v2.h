#ifndef MATMUL_WMMA_V2_H
#define MATMUL_WMMA_V2_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>

// Highly optimized WMMA Tensor Core matmul kernel v2
//
// OPTIMIZATIONS:
// 1. Large block tiles (256x128) with 8 warps
// 2. Deep K tile (64) to maximize compute-to-memory ratio
// 3. Double buffering with async memory copies
// 4. Vectorized 128-bit global memory loads
// 5. Swizzled shared memory layout to eliminate bank conflicts
// 6. Software pipelining
//
// PERFORMANCE TARGET: >30% MFU on H100
//
// REQUIREMENTS:
// - SM80+ (Ampere or newer for cp.async)
// - N must be multiple of 256

class MatmulWmmaV2 : public MatmulKernel {
private:
    int N;
    half *d_A_fp16;
    half *d_B_fp16;

public:
    MatmulWmmaV2(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaV2() override;
};

#endif
