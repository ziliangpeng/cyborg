#ifndef SOFTMAX_CUB_H
#define SOFTMAX_CUB_H

// CUB-based softmax implementation
//
// Uses NVIDIA's CUB (CUDA Unbound) library for optimized block-level reductions.
// CUB is part of the CUDA Core Compute Libraries (CCCL) and provides highly
// optimized primitives that handle warp shuffles, shared memory, and bank conflicts
// automatically across all GPU architectures.
//
// Architecture: 3-kernel approach with CUB optimizations
// - Kernel 1: Block statistics using CUB BlockReduce (max + exp-sum)
// - Kernel 2: Global reduce using CUB BlockReduce (1 block)
// - Kernel 3: Normalize (reuses elementwise kernel)
//
// Benefits over hand-written reductions:
// - Less code: CUB handles shared memory management and synchronization
// - Better performance: NVIDIA-optimized for all GPU architectures
// - More maintainable: Template-based, type-safe interface
// - Portable: Works optimally on Pascal, Volta, Turing, Ampere, Hopper
//
// CUB BlockReduce Features:
// - Automatically selects optimal algorithm (warp shuffle vs shared memory)
// - Handles bank conflicts and memory coalescing
// - Provides Sum(), Max(), and custom reduction operators
// - Template specialization for block size at compile time
//
// Performance characteristics:
// - Expected: Similar or slightly better than hand-written fused3
// - CUB overhead is minimal (compile-time template specialization)
// - Memory access patterns are optimal
//
// Requirements:
// - CUDA 11.0+ (CUB is included in CUDA Toolkit)
// - C++11 or later
//
// Returns execution time in milliseconds (currently 0.0f, timing handled by caller)
float softmax_Cub(const float *d_input, float *d_output, int n, int threadsPerBlock);

#endif
