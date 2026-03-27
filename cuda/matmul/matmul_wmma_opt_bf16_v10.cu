#include "matmul_wmma_opt_bf16_v10.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Tuned for register usage and instruction mix
// ============================================================================

// Same as V3 but with BK=32 for better amortization
#define BM 128
#define BN 128
#define BK 32           // Larger K for better compute/load ratio

#define WM 32
#define WN 64

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M (BM / WM)   // 4
#define WARPS_N (BN / WN)   // 2
#define NUM_WARPS (WARPS_M * WARPS_N)  // 8

#define WMMA_TILES_M (WM / WMMA_M)  // 2
#define WMMA_TILES_N (WN / WMMA_N)  // 4

#define BLOCK_SIZE (NUM_WARPS * 32)  // 256 threads

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16OptV10(const float* input, __nv_bfloat16* output, int size) {
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
// Optimized BF16 WMMA Kernel V10 - Better instruction scheduling
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
matmulWmmaOptBf16V10Kernel(const __nv_bfloat16* __restrict__ A,
                           const __nv_bfloat16* __restrict__ B,
                           float* __restrict__ C,
                           int N) {
#if __CUDA_ARCH__ >= 800
    __shared__ __nv_bfloat16 As[2][BK][BM + 8];
    __shared__ __nv_bfloat16 Bs[2][BK][BN + 8];

    const int warpId = threadIdx.x / 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Accumulators - keep in registers across all K iterations
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    #pragma unroll
    for (int wm = 0; wm < WMMA_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_TILES_N; wn++) {
            wmma::fill_fragment(c_frag[wm][wn], 0.0f);
        }
    }

    // Precompute warp's load offsets
    const int warpABaseRow = warpM * WM;
    const int warpBBaseCol = warpN * WN;

    // Load config
    constexpr int A_LOADS = (BM * BK) / BLOCK_SIZE;  // 16
    constexpr int B_LOADS = (BK * BN) / BLOCK_SIZE;  // 16

    const int numKTiles = N / BK;
    int bufIdx = 0;

    // Load first tile
    #pragma unroll
    for (int i = 0; i < A_LOADS; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BK;
        int col = linearIdx % BK;
        int gRow = blockM + row;
        int gCol = col;

        __nv_bfloat16 val = (gRow < N && gCol < N) ?
            A[gRow * N + gCol] : __float2bfloat16(0.0f);
        As[bufIdx][col][row] = val;
    }

    #pragma unroll
    for (int i = 0; i < B_LOADS; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BN;
        int col = linearIdx % BN;
        int gRow = row;
        int gCol = blockN + col;

        __nv_bfloat16 val = (gRow < N && gCol < N) ?
            B[gRow * N + gCol] : __float2bfloat16(0.0f);
        Bs[bufIdx][row][col] = val;
    }

    __syncthreads();

    // Main loop
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readBuf = bufIdx;
        bufIdx = 1 - bufIdx;

        // Start prefetching next tile BEFORE computing
        if (kTile + 1 < numKTiles) {
            int kOffset = (kTile + 1) * BK;

            #pragma unroll
            for (int i = 0; i < A_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BK;
                int col = linearIdx % BK;
                int gRow = blockM + row;
                int gCol = kOffset + col;

                __nv_bfloat16 val = (gRow < N && gCol < N) ?
                    A[gRow * N + gCol] : __float2bfloat16(0.0f);
                As[bufIdx][col][row] = val;
            }

            #pragma unroll
            for (int i = 0; i < B_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BN;
                int col = linearIdx % BN;
                int gRow = kOffset + row;
                int gCol = blockN + col;

                __nv_bfloat16 val = (gRow < N && gCol < N) ?
                    B[gRow * N + gCol] : __float2bfloat16(0.0f);
                Bs[bufIdx][row][col] = val;
            }
        }

        // Process BK=32 with 2 WMMA_K steps
        // Fully unroll the inner computation
        {
            // K step 0
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
                a_frag0[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
                b_frag0[WMMA_TILES_N];

            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                wmma::load_matrix_sync(a_frag0[wm], &As[readBuf][0][warpABaseRow + wm * WMMA_M], BM + 8);
            }
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                wmma::load_matrix_sync(b_frag0[wn], &Bs[readBuf][0][warpBBaseCol + wn * WMMA_N], BN + 8);
            }

            // Start loading K step 1 fragments while computing K step 0
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
                a_frag1[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
                b_frag1[WMMA_TILES_N];

            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                wmma::load_matrix_sync(a_frag1[wm], &As[readBuf][WMMA_K][warpABaseRow + wm * WMMA_M], BM + 8);
            }
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                wmma::load_matrix_sync(b_frag1[wn], &Bs[readBuf][WMMA_K][warpBBaseCol + wn * WMMA_N], BN + 8);
            }

            // Compute K step 0
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                    wmma::mma_sync(c_frag[wm][wn], a_frag0[wm], b_frag0[wn], c_frag[wm][wn]);
                }
            }

            // Compute K step 1
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                    wmma::mma_sync(c_frag[wm][wn], a_frag1[wm], b_frag1[wn], c_frag[wm][wn]);
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
            int cRow = blockM + warpABaseRow + wm * WMMA_M;
            int cCol = blockN + warpBBaseCol + wn * WMMA_N;

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

MatmulWmmaOptBf16V10::MatmulWmmaOptBf16V10(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("BF16 WMMA requires SM 8.0+");
    }

    if (N % 128 != 0) {
        throw std::runtime_error("N must be multiple of 128");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulWmmaOptBf16V10::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16OptV10<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16OptV10<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulWmmaOptBf16V10Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWmmaOptBf16V10::~MatmulWmmaOptBf16V10() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
