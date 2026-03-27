#ifndef MATMUL_WMMA_V3_H
#define MATMUL_WMMA_V3_H

#include "matmul_kernel.h"
#include <cuda_fp16.h>

// WMMA Tensor Core kernel V3 - Balanced optimization
//
// Key insight: V1 was faster than V2 because:
// - 256 threads (8 warps) has better occupancy than 512 (16 warps)
// - Smaller shared memory footprint allows more blocks per SM
//
// V3 improvements over V1:
// 1. Keep 256 threads (8 warps) configuration
// 2. Increase BK to 64 for better compute/load ratio
// 3. Use swizzled indexing to eliminate bank conflicts
// 4. Pre-compute load indices to reduce register pressure
//
// Target: >20% MFU

class MatmulWmmaV3 : public MatmulKernel {
private:
    int N;
    half *d_A_fp16;
    half *d_B_fp16;

public:
    MatmulWmmaV3(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWmmaV3() override;
};

#endif
