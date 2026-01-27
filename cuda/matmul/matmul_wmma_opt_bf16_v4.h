#ifndef MATMUL_WMMA_OPT_BF16_V4_H
#define MATMUL_WMMA_OPT_BF16_V4_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V4
//
// Advanced optimizations:
// 1. cp.async for asynchronous global to shared memory copies
// 2. Multi-stage software pipelining (3 stages)
// 3. Aggressive register blocking

class MatmulWmmaOptBf16V4 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16V4(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V4() override;
};

#endif
