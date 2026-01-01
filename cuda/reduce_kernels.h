#ifndef REDUCE_KERNELS_H
#define REDUCE_KERNELS_H

// Option B: Fully GPU recursive reduction (reduces until 1 element)
float vectorSum_GPU(const float *d_input, int n, int threadsPerBlock);

// Option D: GPU with configurable CPU threshold
float vectorSum_Threshold(const float *d_input, int n, int threadsPerBlock, int cpuThreshold);

// Warp-optimized variants (use warp shuffles for final 32→1 reduction)
float vectorSum_GPU_Warp(const float *d_input, int n, int threadsPerBlock);
float vectorSum_Threshold_Warp(const float *d_input, int n, int threadsPerBlock, int cpuThreshold);

#endif
