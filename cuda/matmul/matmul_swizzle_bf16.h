#ifndef MATMUL_SWIZZLE_BF16_H
#define MATMUL_SWIZZLE_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// BF16 matmul with swizzled shared memory layout
//
// Uses XOR-based swizzle pattern to eliminate bank conflicts
// without wasting memory on padding

class MatmulSwizzleBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulSwizzleBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulSwizzleBf16() override;
};

#endif
