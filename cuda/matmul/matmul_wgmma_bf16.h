#ifndef MATMUL_WGMMA_BF16_H
#define MATMUL_WGMMA_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// EXPERIMENTAL: WGMMA BF16 matmul kernel using direct PTX
// Uses wgmma.mma_async.sync.aligned instructions for H100
// Requires SM90a (Hopper architecture with accelerated features)
//
// NOTE: This is a work-in-progress. WGMMA requires specific swizzled
// memory layouts that are not yet implemented. The current implementation
// compiles and runs but produces incorrect results. For production use,
// consider using CUTLASS which handles the complex memory layouts.

class MatmulWgmmaBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulWgmmaBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWgmmaBf16() override;
};

#endif
