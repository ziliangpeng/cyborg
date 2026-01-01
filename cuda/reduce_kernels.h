#ifndef REDUCE_KERNELS_H
#define REDUCE_KERNELS_H

// Option B: Fully GPU recursive reduction (reduces until 1 element)
// Returns the final sum value
float vectorSum_GPU(const float *d_input, int n, int threadsPerBlock);

// Option D: GPU with configurable CPU threshold
// Reduces on GPU until size <= cpuThreshold, then finishes on CPU
float vectorSum_Threshold(const float *d_input, int n, int threadsPerBlock, int cpuThreshold);

#endif
