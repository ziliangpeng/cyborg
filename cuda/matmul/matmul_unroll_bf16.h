#ifndef MATMUL_UNROLL_BF16_H
#define MATMUL_UNROLL_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Unrolled BF16 matmul kernel
// Process multiple K tiles per iteration to reduce sync overhead
// Uses 4x K unrolling with register-level fragment reuse

class MatmulUnrollBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulUnrollBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulUnrollBf16() override;
};

#endif
