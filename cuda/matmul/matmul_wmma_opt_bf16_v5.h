#ifndef MATMUL_WMMA_OPT_BF16_V5_H
#define MATMUL_WMMA_OPT_BF16_V5_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V5
//
// Focus on maximizing arithmetic intensity:
// 1. Larger block tiles (256x256) with careful occupancy
// 2. Each warp computes 128x64 output (8x4 WMMA tiles)
// 3. Minimize shared memory traffic

class MatmulWmmaOptBf16V5 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16V5(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V5() override;
};

#endif
