#ifndef SOFTMAX_KERNELS_H
#define SOFTMAX_KERNELS_H

// Naive (unstable) - for demonstration of overflow issues
// Returns execution time in milliseconds
float softmax_Naive(const float *d_input, float *d_output, int n, int threadsPerBlock);

// Multi-pass (3 kernels) - stable baseline approach
// Stage 1: Find max, Stage 2: Compute exp-sum, Stage 3: Normalize
// Returns execution time in milliseconds
float softmax_MultiPass(const float *d_input, float *d_output, int n, int threadsPerBlock);

// Fused (1-2 kernels) - optimized with kernel fusion (SKELETON - not implemented)
// Returns execution time in milliseconds
float softmax_Fused(const float *d_input, float *d_output, int n, int threadsPerBlock);

// Online algorithm - single pass streaming computation (SKELETON - not implemented)
// Returns execution time in milliseconds
float softmax_Online(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
