#ifndef MATMUL_WMMA_OPT_BF16_V3_H
#define MATMUL_WMMA_OPT_BF16_V3_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V3
//
// Focus on maximizing tensor core utilization:
// 1. Smaller BK (16) to reduce shared memory and improve occupancy
// 2. More WMMA operations per shared memory load
// 3. 128x128 block tiles with 8 warps

class MatmulWmmaOptBf16V3 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16V3(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V3() override;
};

#endif
