#ifndef MATMUL_HYBRID_BF16_H
#define MATMUL_HYBRID_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Hybrid BF16 matmul kernel
// Combines:
// - 3-stage pipelining
// - 2x K unrolling within each stage
// - RLRL register reuse pattern

class MatmulHybridBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulHybridBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulHybridBf16() override;
};

#endif
