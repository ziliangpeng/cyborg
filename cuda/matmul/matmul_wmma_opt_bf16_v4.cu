#include "matmul_wmma_opt_bf16_v4.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Tuned for H100
// ============================================================================

#define BM 128          // Block tile M
#define BN 256          // Block tile N (wider for more parallelism)
#define BK 32           // K tile

#define WM 64           // Warp tile M
#define WN 64           // Warp tile N

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M (BM / WM)   // 2
#define WARPS_N (BN / WN)   // 4
#define NUM_WARPS (WARPS_M * WARPS_N)  // 8

#define WMMA_TILES_M (WM / WMMA_M)  // 4
#define WMMA_TILES_N (WN / WMMA_N)  // 4

#define BLOCK_SIZE (NUM_WARPS * 32)  // 256 threads

#define NUM_STAGES 3  // Pipeline stages

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16OptV4(const float* input, __nv_bfloat16* output, int size) {
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
// Optimized BF16 WMMA Kernel V4 with cp.async
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
matmulWmmaOptBf16V4Kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          float* __restrict__ C,
                          int N) {
#if __CUDA_ARCH__ >= 800
    // Multi-stage shared memory buffers
    // A: BM x BK stored as BK x BM (transposed for col_major)
    // B: BK x BN stored as row_major
    __shared__ __nv_bfloat16 As[NUM_STAGES][BK][BM + 8];
    __shared__ __nv_bfloat16 Bs[NUM_STAGES][BK][BN + 8];

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Accumulators - each warp computes 4x4 WMMA tiles = 64x64 output
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    #pragma unroll
    for (int wm = 0; wm < WMMA_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_TILES_N; wn++) {
            wmma::fill_fragment(c_frag[wm][wn], 0.0f);
        }
    }

    // Load configuration
    // A: BM * BK = 128 * 32 = 4096 elements, 256 threads -> 16 per thread
    // B: BK * BN = 32 * 256 = 8192 elements, 256 threads -> 32 per thread
    constexpr int A_LOADS = (BM * BK) / BLOCK_SIZE;  // 16
    constexpr int B_LOADS = (BK * BN) / BLOCK_SIZE;  // 32

    const int numKTiles = N / BK;

    // Fill pipeline with initial stages
    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES - 1 && stage < numKTiles; stage++) {
        int kOffset = stage * BK;

        // Load A tile (transposed)
        #pragma unroll
        for (int i = 0; i < A_LOADS; i++) {
            int linearIdx = threadIdx.x + i * BLOCK_SIZE;
            int row = linearIdx / BK;
            int col = linearIdx % BK;

            __nv_bfloat16 val = (blockM + row < N && kOffset + col < N) ?
                A[(blockM + row) * N + kOffset + col] : __float2bfloat16(0.0f);
            As[stage][col][row] = val;
        }

        // Load B tile
        #pragma unroll
        for (int i = 0; i < B_LOADS; i++) {
            int linearIdx = threadIdx.x + i * BLOCK_SIZE;
            int row = linearIdx / BN;
            int col = linearIdx % BN;

            __nv_bfloat16 val = (kOffset + row < N && blockN + col < N) ?
                B[(kOffset + row) * N + blockN + col] : __float2bfloat16(0.0f);
            Bs[stage][row][col] = val;
        }

        __pipeline_commit();
    }

    // Main loop with software pipelining
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readStage = kTile % NUM_STAGES;
        int writeStage = (kTile + NUM_STAGES - 1) % NUM_STAGES;

        // Wait for current stage to be ready
        __pipeline_wait_prior(NUM_STAGES - 2);
        __syncthreads();

        // Prefetch next stage
        if (kTile + NUM_STAGES - 1 < numKTiles) {
            int kOffset = (kTile + NUM_STAGES - 1) * BK;

            #pragma unroll
            for (int i = 0; i < A_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BK;
                int col = linearIdx % BK;

                __nv_bfloat16 val = (blockM + row < N && kOffset + col < N) ?
                    A[(blockM + row) * N + kOffset + col] : __float2bfloat16(0.0f);
                As[writeStage][col][row] = val;
            }

            #pragma unroll
            for (int i = 0; i < B_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BN;
                int col = linearIdx % BN;

                __nv_bfloat16 val = (kOffset + row < N && blockN + col < N) ?
                    B[(kOffset + row) * N + blockN + col] : __float2bfloat16(0.0f);
                Bs[writeStage][row][col] = val;
            }

            __pipeline_commit();
        }

        // Compute: process BK/WMMA_K = 2 k-steps
        #pragma unroll
        for (int k = 0; k < BK / WMMA_K; k++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
                a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
                b_frag[WMMA_TILES_N];

            // Load A fragments for this warp
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                int aRow = warpM * WM + wm * WMMA_M;
                int aK = k * WMMA_K;
                wmma::load_matrix_sync(a_frag[wm], &As[readStage][aK][aRow], BM + 8);
            }

            // Load B fragments for this warp
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                int bCol = warpN * WN + wn * WMMA_N;
                int bK = k * WMMA_K;
                wmma::load_matrix_sync(b_frag[wn], &Bs[readStage][bK][bCol], BN + 8);
            }

            // Compute all 4x4 WMMA tiles
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                    wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int wm = 0; wm < WMMA_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_TILES_N; wn++) {
            int cRow = blockM + warpM * WM + wm * WMMA_M;
            int cCol = blockN + warpN * WN + wn * WMMA_N;

            if (cRow + WMMA_M <= N && cCol + WMMA_N <= N) {
                wmma::store_matrix_sync(&C[cRow * N + cCol], c_frag[wm][wn], N, wmma::mem_row_major);
            }
        }
    }
#endif
}

// ============================================================================
// Class Implementation
// ============================================================================

MatmulWmmaOptBf16V4::MatmulWmmaOptBf16V4(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("BF16 WMMA requires SM 8.0+");
    }

    if (N % 256 != 0) {
        throw std::runtime_error("N must be multiple of 256 for V4");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulWmmaOptBf16V4::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16OptV4<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16OptV4<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulWmmaOptBf16V4Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWmmaOptBf16V4::~MatmulWmmaOptBf16V4() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
