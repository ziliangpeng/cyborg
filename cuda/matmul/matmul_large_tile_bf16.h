#ifndef MATMUL_LARGE_TILE_BF16_H
#define MATMUL_LARGE_TILE_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// BF16 matmul with large tiles and register reuse
//
// Configuration based on Bruce-Lee-LY's optimal settings:
// - 256x128 block tile
// - 64x64 warp tile
// - 4x4 WMMA tiles per warp
// - Register-level A matrix reuse ("RLRL" pattern)

class MatmulLargeTileBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulLargeTileBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulLargeTileBf16() override;
};

#endif
