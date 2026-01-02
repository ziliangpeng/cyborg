#ifndef SOFTMAX_FUSED_H
#define SOFTMAX_FUSED_H

// Fused 3-kernel softmax - optimized stable approach
//
// Block-level fusion with 3 kernel launches:
// Kernel 1: Block statistics (max + exp-sum in single pass)
// Kernel 2: Global reduce to merge block statistics
// Kernel 3: Final normalization
//
// 2.15x faster than multi-pass for 1M elements by eliminating
// recursive kernel launches.
//
// Returns execution time in milliseconds
float softmax_Fused(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
