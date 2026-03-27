#ifndef MATMUL_WMMA_OPT_BF16_V7_H
#define MATMUL_WMMA_OPT_BF16_V7_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V7
//
// Large warp tiles with maximum compute per shared memory load:
// 1. 128x128 block tiles with 4 warps
// 2. Each warp computes 64x64 (4x4 WMMA tiles) - maximum compute density
// 3. Larger BK=32 to amortize loop overhead
// 4. Focus on maximizing arithmetic intensity

class MatmulWmmaOptBf16V7 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16V7(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V7() override;
};

#endif
