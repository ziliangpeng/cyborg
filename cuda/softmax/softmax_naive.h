#ifndef SOFTMAX_NAIVE_H
#define SOFTMAX_NAIVE_H

// Naive (unstable) softmax - for demonstration of overflow issues
//
// This implementation computes exp(x) directly without max subtraction,
// which causes numerical overflow for large input values.
//
// Returns execution time in milliseconds
float softmax_Naive(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
