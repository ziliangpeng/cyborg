#ifndef SOFTMAX_FUSED2_H
#define SOFTMAX_FUSED2_H

// Fused 2-kernel softmax - merge global reduce + normalize (SKELETON - not implemented)
//
// Goal: Merge Kernel 2 (global reduce) + Kernel 3 (normalize) from the 3-kernel version
// Expected: 10-20% faster than 3-kernel by eliminating one kernel launch
//
// Kernel 1: Block statistics (reuse from 3-kernel)
// Kernel 2: Fused global reduce + normalize in single pass
//
// Returns execution time in milliseconds
float softmax_Fused2(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
