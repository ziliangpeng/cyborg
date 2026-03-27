#ifndef MATMUL_BEST_BF16_H
#define MATMUL_BEST_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// Best effort BF16 matmul kernel
// Combines all optimizations:
// - 128x128 block tiles
// - 64x64 warp tiles (maximize WMMA tiles per warp)
// - 4x K unrolling
// - RLRL register reuse pattern
// - Optimized shared memory layout

class MatmulBestBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulBestBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulBestBf16() override;
};

#endif
