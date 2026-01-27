#ifndef MATMUL_STAGE3_BF16_H
#define MATMUL_STAGE3_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// BF16 matmul with 3-stage software pipelining
// Based on CUTLASS's approach - uses WMMA with proper staging

class MatmulStage3Bf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulStage3Bf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulStage3Bf16() override;
};

#endif
