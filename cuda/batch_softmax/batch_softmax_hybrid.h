#ifndef BATCH_SOFTMAX_HYBRID_H
#define BATCH_SOFTMAX_HYBRID_H

#include "batch_softmax_kernel.h"

// Hybrid batch softmax with adaptive kernel selection
//
// Automatically selects the optimal kernel based on dimension size:
// - dim <= 64:   Warp kernel (32 threads) - minimal overhead
// - dim <= 1024: Multi-warp kernel (256 threads) - balanced
// - dim > 1024:  Multi-warp with vectorized loads - memory bandwidth optimized
//
// This provides a single interface that performs well across all dimension
// sizes without requiring the user to manually select the best kernel.
//
// Selection rationale:
// - Small dims (<=64): Each thread processes <=2 elements, warp kernel is optimal
//   due to zero shared memory and minimal synchronization
// - Medium dims (<=1024): More threads help with instruction-level parallelism,
//   multi-warp hybrid reduction is efficient
// - Large dims (>1024): Memory bandwidth becomes the bottleneck, vectorized
//   loads provide significant speedup
//
// The thresholds are tuned for typical GPU architectures (Volta, Ampere, Hopper).
// Actual optimal crossover points may vary by GPU model.
class HybridBatchSoftmax : public BatchSoftmaxKernel {
public:
    HybridBatchSoftmax(int batch_size, int dim, int threadsPerBlock);
    void execute(const float *d_input, float *d_output) override;
    ~HybridBatchSoftmax() = default;

private:
    int batch_size;
    int dim;
    int selected_kernel;  // 0=warp, 1=multi_warp_scalar, 2=multi_warp_vec4
};

#endif  // BATCH_SOFTMAX_HYBRID_H
