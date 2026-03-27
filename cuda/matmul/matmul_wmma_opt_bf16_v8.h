#ifndef MATMUL_WMMA_OPT_BF16_V8_H
#define MATMUL_WMMA_OPT_BF16_V8_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V8
//
// Key optimizations:
// 1. Same configuration as V3 (our best: 128x128, BK=16, 8 warps)
// 2. Vectorized BF16 loads using float4 for 4x bf16
// 3. Swizzled shared memory layout to avoid bank conflicts
// 4. Interleaved loading and computing patterns

class MatmulWmmaOptBf16V8 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16V8(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V8() override;
};

#endif
