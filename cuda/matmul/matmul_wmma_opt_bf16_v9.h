#ifndef MATMUL_WMMA_OPT_BF16_V9_H
#define MATMUL_WMMA_OPT_BF16_V9_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V9
//
// Split-K parallelization:
// 1. Divides K dimension into multiple parallel chunks
// 2. Each threadblock computes partial results
// 3. Final reduction to get complete output
// This improves SM utilization for smaller matrix dimensions

class MatmulWmmaOptBf16V9 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;
    float *d_workspace;  // For partial results

public:
    MatmulWmmaOptBf16V9(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V9() override;
};

#endif
