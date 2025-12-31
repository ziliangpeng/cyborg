#ifndef VECTOR_KERNELS_H
#define VECTOR_KERNELS_H

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n);

#endif
