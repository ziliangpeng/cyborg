#ifndef MATMUL_ASYNC_BF16_H
#define MATMUL_ASYNC_BF16_H

#include "matmul_kernel.h"
#include <cuda_bf16.h>

// BF16 matmul with cp.async and proper software pipelining
//
// Key optimizations:
// 1. cp.async for asynchronous global to shared memory copies
// 2. 3-stage software pipelining
// 3. Swizzled shared memory layout
// 4. WMMA for tensor core operations

class MatmulAsyncBf16 : public MatmulKernel {
private:
    int N;
    __nv_bfloat16 *d_A_bf16;
    __nv_bfloat16 *d_B_bf16;

public:
    MatmulAsyncBf16(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulAsyncBf16() override;
};

#endif
