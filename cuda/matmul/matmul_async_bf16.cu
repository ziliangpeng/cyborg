#include "matmul_async_bf16.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Based on Bruce-Lee-LY's optimal settings
// ============================================================================

#define BM 128          // Block tile M
#define BN 128          // Block tile N
#define BK 32           // K tile - larger for cp.async efficiency

#define WM 64           // Warp tile M
#define WN 32           // Warp tile N

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M (BM / WM)   // 2
#define WARPS_N (BN / WN)   // 4
#define NUM_WARPS (WARPS_M * WARPS_N)  // 8

// Each warp computes 4x2 = 8 WMMA tiles (64x32 output)
#define WMMA_TILES_M (WM / WMMA_M)  // 4
#define WMMA_TILES_N (WN / WMMA_N)  // 2

#define BLOCK_SIZE (NUM_WARPS * 32)  // 256 threads

#define NUM_STAGES 3

// Swizzle XOR mask for bank conflict avoidance
#define SWIZZLE_MASK 7  // XOR with bits [2:0]

// ============================================================================
// Helper: Swizzled address calculation
// ============================================================================

__device__ __forceinline__ int swizzle_offset(int row, int col, int stride) {
    // XOR row index with part of column to spread across banks
    int swizzled_row = row ^ ((col >> 4) & SWIZZLE_MASK);
    return swizzled_row * stride + col;
}

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16Async(const float* input, __nv_bfloat16* output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 in4 = *reinterpret_cast<const float4*>(&input[idx]);
        __nv_bfloat162 h0 = __floats2bfloat162_rn(in4.x, in4.y);
        __nv_bfloat162 h1 = __floats2bfloat162_rn(in4.z, in4.w);
        *reinterpret_cast<__nv_bfloat162*>(&output[idx]) = h0;
        *reinterpret_cast<__nv_bfloat162*>(&output[idx + 2]) = h1;
    }
}

// ============================================================================
// Async Copy Kernel with 3-stage Pipeline
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
matmulAsyncBf16Kernel(const __nv_bfloat16* __restrict__ A,
                      const __nv_bfloat16* __restrict__ B,
                      float* __restrict__ C,
                      int N) {
#if __CUDA_ARCH__ >= 800
    // Shared memory: 3 stages for A and B
    // A stored as col-major (transposed) for efficient WMMA loads
    // Using padding (+16) instead of swizzle for simplicity with WMMA
    __shared__ __nv_bfloat16 As[NUM_STAGES][BK][BM + 16];
    __shared__ __nv_bfloat16 Bs[NUM_STAGES][BK][BN + 16];

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Accumulators for each warp's WMMA tiles
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    #pragma unroll
    for (int m = 0; m < WMMA_TILES_M; m++) {
        #pragma unroll
        for (int n = 0; n < WMMA_TILES_N; n++) {
            wmma::fill_fragment(c_frag[m][n], 0.0f);
        }
    }

    // Calculate per-thread load assignments
    // Using cp.async with 16-byte (8 BF16) granularity
    // A: BM * BK = 128 * 32 = 4096 BF16 = 512 16-byte chunks
    // B: BK * BN = 32 * 128 = 4096 BF16 = 512 16-byte chunks
    // 256 threads -> 2 16-byte loads per thread for each matrix

    const int numKTiles = N / BK;

    // Prologue: fill pipeline stages 0 to NUM_STAGES-2
    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES - 1; stage++) {
        if (stage < numKTiles) {
            int kOffset = stage * BK;

            // Load A: each thread loads 16 BF16 elements (two 8-element chunks)
            // We need to load BM * BK = 4096 elements with 256 threads
            // = 16 elements per thread
            #pragma unroll
            for (int load = 0; load < 2; load++) {
                // Calculate source and destination for this load
                int elemIdx = (threadIdx.x * 2 + load) * 8;  // 8 BF16 per cp.async
                int row = elemIdx % BM;
                int col = elemIdx / BM;

                if (col < BK) {
                    int gRow = blockM + row;
                    int gCol = kOffset + col;

                    // Source address in global memory
                    const __nv_bfloat16* src = &A[gRow * N + gCol];

                    // Destination in shared memory (transposed)
                    __nv_bfloat16* dst = &As[stage][col][row];

                    // Issue cp.async
                    if (gRow < N && gCol + 7 < N) {
                        __pipeline_memcpy_async(dst, src, 16);  // 16 bytes = 8 BF16
                    } else {
                        // Handle boundary - zero fill
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            dst[i] = (gRow < N && gCol + i < N) ?
                                A[gRow * N + gCol + i] : __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            // Load B: same pattern
            #pragma unroll
            for (int load = 0; load < 2; load++) {
                int elemIdx = (threadIdx.x * 2 + load) * 8;
                int row = elemIdx / BN;
                int col = elemIdx % BN;

                if (row < BK) {
                    int gRow = kOffset + row;
                    int gCol = blockN + col;

                    const __nv_bfloat16* src = &B[gRow * N + gCol];
                    __nv_bfloat16* dst = &Bs[stage][row][col];

                    if (gRow < N && gCol + 7 < N) {
                        __pipeline_memcpy_async(dst, src, 16);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            dst[i] = (gRow < N && gCol + i < N) ?
                                B[gRow * N + gCol + i] : __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            __pipeline_commit();
        }
    }

    // Main loop
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readStage = kTile % NUM_STAGES;
        int writeStage = (kTile + NUM_STAGES - 1) % NUM_STAGES;

        // Wait for current stage to be ready
        __pipeline_wait_prior(NUM_STAGES - 2);
        __syncthreads();

        // Issue loads for next stage
        if (kTile + NUM_STAGES - 1 < numKTiles) {
            int kOffset = (kTile + NUM_STAGES - 1) * BK;

            #pragma unroll
            for (int load = 0; load < 2; load++) {
                int elemIdx = (threadIdx.x * 2 + load) * 8;
                int row = elemIdx % BM;
                int col = elemIdx / BM;

                if (col < BK) {
                    int gRow = blockM + row;
                    int gCol = kOffset + col;

                    const __nv_bfloat16* src = &A[gRow * N + gCol];
                    __nv_bfloat16* dst = &As[writeStage][col][row];

                    if (gRow < N && gCol + 7 < N) {
                        __pipeline_memcpy_async(dst, src, 16);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            dst[i] = (gRow < N && gCol + i < N) ?
                                A[gRow * N + gCol + i] : __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            #pragma unroll
            for (int load = 0; load < 2; load++) {
                int elemIdx = (threadIdx.x * 2 + load) * 8;
                int row = elemIdx / BN;
                int col = elemIdx % BN;

                if (row < BK) {
                    int gRow = kOffset + row;
                    int gCol = blockN + col;

                    const __nv_bfloat16* src = &B[gRow * N + gCol];
                    __nv_bfloat16* dst = &Bs[writeStage][row][col];

                    if (gRow < N && gCol + 7 < N) {
                        __pipeline_memcpy_async(dst, src, 16);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            dst[i] = (gRow < N && gCol + i < N) ?
                                B[gRow * N + gCol + i] : __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            __pipeline_commit();
        }

        // Compute on current stage
        // Process BK/WMMA_K = 2 k-steps
        #pragma unroll
        for (int k = 0; k < BK / WMMA_K; k++) {
            // Load WMMA fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
                a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
                b_frag[WMMA_TILES_N];

            // Load A fragments for this warp
            #pragma unroll
            for (int m = 0; m < WMMA_TILES_M; m++) {
                int aRow = warpM * WM + m * WMMA_M;
                int aK = k * WMMA_K;
                wmma::load_matrix_sync(a_frag[m], &As[readStage][aK][aRow], BM + 16);
            }

            // Load B fragments for this warp
            #pragma unroll
            for (int n = 0; n < WMMA_TILES_N; n++) {
                int bCol = warpN * WN + n * WMMA_N;
                int bK = k * WMMA_K;
                wmma::load_matrix_sync(b_frag[n], &Bs[readStage][bK][bCol], BN + 16);
            }

            // Compute 4x2 WMMA operations
            #pragma unroll
            for (int m = 0; m < WMMA_TILES_M; m++) {
                #pragma unroll
                for (int n = 0; n < WMMA_TILES_N; n++) {
                    wmma::mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int m = 0; m < WMMA_TILES_M; m++) {
        #pragma unroll
        for (int n = 0; n < WMMA_TILES_N; n++) {
            int cRow = blockM + warpM * WM + m * WMMA_M;
            int cCol = blockN + warpN * WN + n * WMMA_N;

            if (cRow + WMMA_M <= N && cCol + WMMA_N <= N) {
                wmma::store_matrix_sync(&C[cRow * N + cCol], c_frag[m][n], N, wmma::mem_row_major);
            }
        }
    }
#endif
}

// ============================================================================
// Class Implementation
// ============================================================================

MatmulAsyncBf16::MatmulAsyncBf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("Async BF16 requires SM 8.0+");
    }

    if (N % 128 != 0) {
        throw std::runtime_error("N must be multiple of 128");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulAsyncBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16Async<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16Async<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulAsyncBf16Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulAsyncBf16::~MatmulAsyncBf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
