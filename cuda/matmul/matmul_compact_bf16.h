#ifndef MATMUL_COMPACT_BF16_H
#define MATMUL_COMPACT_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Compact BF16 matmul kernel
// Focus on:
// - Smaller shared memory footprint for higher occupancy
// - 32x32 warp tiles (4 WMMA tiles per warp)
// - 2-stage double buffering
// - BK=8 for even smaller shared memory

class MatmulCompactBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulCompactBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulCompactBf16() override;
};

#endif
