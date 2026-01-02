#ifndef SOFTMAX_CUB_DEVICE_H
#define SOFTMAX_CUB_DEVICE_H

// CUB Device-Level softmax implementation
//
// Uses NVIDIA's CUB (CUDA Unbound) device-level primitives for single-call reductions.
// CUB DeviceReduce provides a single function call that internally handles kernel launches,
// temporary storage allocation, and grid configuration.
//
// Architecture: 2-kernel approach with CUB DeviceReduce
// - Kernel 1: cub::DeviceReduce::Max to find global maximum (CUB handles everything)
// - Kernel 2: Custom kernel for exp-sum + normalize in single pass
//
// Comparison with Block-Level CUB:
// - Block-level (3 kernels): More control, better data locality, typically faster
// - Device-level (2 kernels): Simpler code, less control, relies on CUB's heuristics
//
// Expected Performance:
// - Likely slower than block-level due to:
//   1. Multiple passes over data (less cache-friendly)
//   2. CUB DeviceReduce launches separate kernels internally
//   3. Less opportunity to exploit data locality
//
// Benefits:
// - Very simple code (single function call for reductions)
// - No manual grid configuration needed
// - Good for prototyping and simple use cases
//
// Requirements:
// - CUDA 11.0+ (CUB is included in CUDA Toolkit)
// - C++11 or later
//
// Returns execution time in milliseconds (currently 0.0f, timing handled by caller)
float softmax_CubDevice(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
