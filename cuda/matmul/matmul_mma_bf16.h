#ifndef MATMUL_MMA_BF16_H
#define MATMUL_MMA_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// High-performance BF16 matmul using PTX MMA instructions
//
// Key optimizations:
// 1. PTX mma.sync.m16n8k16 instead of WMMA API
// 2. ldmatrix for efficient shared memory to register transfer
// 3. Swizzled shared memory layout to avoid bank conflicts
// 4. 3-stage software pipelining with cp.async
// 5. Register-level double buffering

class MatmulMmaBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulMmaBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulMmaBf16() override;
};

#endif
