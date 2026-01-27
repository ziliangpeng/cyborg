#ifndef MATMUL_WMMA_OPT_BF16_V2_H
#define MATMUL_WMMA_OPT_BF16_V2_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V2
//
// Improvements over V1:
// 1. Larger tiles (256x256 block, 128 K)
// 2. More warps (16 warps = 512 threads)
// 3. Async memory copies using cp.async
//
// Requires SM80+

class MatmulWmmaOptBf16V2 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16V2(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V2() override;
};

#endif
