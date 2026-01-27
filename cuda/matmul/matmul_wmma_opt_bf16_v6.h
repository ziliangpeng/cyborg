#ifndef MATMUL_WMMA_OPT_BF16_V6_H
#define MATMUL_WMMA_OPT_BF16_V6_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V6
//
// Optimizations:
// 1. Smaller 64x64 block tiles for better occupancy
// 2. 4 warps (128 threads) - matches natural WMMA tile arrangement
// 3. Each warp computes 32x32 (2x2 WMMA tiles)
// 4. Minimal shared memory for maximum occupancy

class MatmulWmmaOptBf16V6 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16V6(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V6() override;
};

#endif
