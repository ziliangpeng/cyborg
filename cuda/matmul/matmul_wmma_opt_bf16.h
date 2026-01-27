#ifndef MATMUL_WMMA_OPT_BF16_H
#define MATMUL_WMMA_OPT_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA Tensor Core matmul kernel using BF16
//
// Same optimizations as wmma_opt but using BF16 instead of FP16
// for fair comparison with cuBLAS BF16.
//
// OPTIMIZATIONS:
// 1. Large block tiles (128x256) with shared memory staging
// 2. Double buffering for memory latency hiding
// 3. Multiple WMMA tiles per warp (4x4 = 16 tiles)
// 4. Shared memory padding to avoid bank conflicts
//
// REQUIREMENTS:
// - SM80+ (Ampere or newer for BF16 WMMA)
// - N must be multiple of 256

class MatmulWmmaOptBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16() override;
};

#endif
