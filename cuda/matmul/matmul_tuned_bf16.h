#ifndef MATMUL_TUNED_BF16_H
#define MATMUL_TUNED_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Tuned BF16 matmul kernel
// Combines best techniques:
// - 128x256 block tiles (wider N for coalescing)
// - BK=16 for occupancy
// - 64x64 warp tiles
// - Double buffering with register-level prefetch

class MatmulTunedBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulTunedBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulTunedBf16() override;
};

#endif
