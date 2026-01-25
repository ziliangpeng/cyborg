#include "batch_softmax_hybrid.h"
#include "batch_softmax_warp.h"
#include "batch_softmax_multi_warp.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

// ============================================================================
// HYBRID BATCH SOFTMAX: Adaptive Kernel Selection
// ============================================================================
//
// This implementation provides a single interface that automatically selects
// the optimal kernel based on the dimension size at construction time.
//
// KERNEL SELECTION STRATEGY:
// --------------------------
// dim <= 64:    Warp kernel (32 threads)
//   - Each thread processes <=2 elements
//   - Zero shared memory, zero __syncthreads()
//   - Minimal launch overhead
//   - Best: Low latency for small dims
//
// dim > 64:     Multi-warp kernel (256 threads, with vectorization if dim % 4 == 0)
//   - 8 warps = more parallelism
//   - Hybrid reduction (warp shuffles + minimal shared mem)
//   - Vectorized loads (float4) for large dims when possible
//   - Best: Medium to large dims where ILP and bandwidth help
//
// IMPLEMENTATION:
// ---------------
// Instead of duplicating kernel code, this class composes the existing
// WarpBatchSoftmax and MultiWarpBatchSoftmax classes internally.
// This ensures consistency and reduces maintenance burden.
//
// ============================================================================

HybridBatchSoftmax::HybridBatchSoftmax(int batch_size, int dim, int threadsPerBlock)
    : batch_size(batch_size), dim(dim) {
    (void)threadsPerBlock;  // Ignored - we select automatically

    // Select kernel based on dimension size
    // dim <= 64: warp kernel is optimal (each thread handles <=2 elements)
    // dim > 64: multi_warp kernel provides better parallelism and vectorization
    if (dim <= 64) {
        selected_kernel = 0;  // Warp kernel
    } else {
        selected_kernel = 1;  // Multi-warp (handles vectorization internally)
    }
}

void HybridBatchSoftmax::execute(const float *d_input, float *d_output) {
    // Dispatch to appropriate kernel
    // Note: We create temporary kernel objects here. The overhead is minimal
    // since constructors just set member variables (no allocations).
    // For production use, consider caching these objects in the class.

    if (selected_kernel == 0) {
        WarpBatchSoftmax kernel(batch_size, dim, 32);
        kernel.execute(d_input, d_output);
    } else {
        MultiWarpBatchSoftmax kernel(batch_size, dim, 256);
        kernel.execute(d_input, d_output);
    }
}
