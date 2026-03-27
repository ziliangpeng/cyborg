#ifndef MATMUL_WMMA_OPT_BF16_V10_H
#define MATMUL_WMMA_OPT_BF16_V10_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Optimized WMMA BF16 kernel V10
//
// Focus on instruction-level optimization:
// 1. Same tile config as V3 (our best baseline)
// 2. Register-resident fragments reused across K iterations
// 3. Manual unrolling for better instruction scheduling
// 4. Prefetch into registers before shared memory sync

class MatmulWmmaOptBf16V10 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWmmaOptBf16V10(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaOptBf16V10() override;
};

#endif
