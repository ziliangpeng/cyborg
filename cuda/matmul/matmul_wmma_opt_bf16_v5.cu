#include "matmul_wmma_opt_bf16_v5.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Focus on arithmetic intensity
// ============================================================================

#define BM 256          // Block tile M
#define BN 128          // Block tile N
#define BK 32           // K tile

#define WM 128          // Warp tile M (large!)
#define WN 32           // Warp tile N

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M (BM / WM)   // 2
#define WARPS_N (BN / WN)   // 4
#define NUM_WARPS (WARPS_M * WARPS_N)  // 8

// Each warp computes 8x2 WMMA tiles = 128x32 output
#define WMMA_TILES_M (WM / WMMA_M)  // 8
#define WMMA_TILES_N (WN / WMMA_N)  // 2

#define BLOCK_SIZE (NUM_WARPS * 32)  // 256 threads

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16OptV5(const float* input, __nv_bfloat16* output, int size) {
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
// Optimized BF16 WMMA Kernel V5 - High arithmetic intensity
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
matmulWmmaOptBf16V5Kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          float* __restrict__ C,
                          int N) {
#if __CUDA_ARCH__ >= 800
    // Double buffered shared memory
    __shared__ __nv_bfloat16 As[2][BK][BM + 8];  // Transposed: [k][m]
    __shared__ __nv_bfloat16 Bs[2][BK][BN + 8];  // Row major: [k][n]

    const int warpId = threadIdx.x / 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Large accumulator array - each warp computes 8x2 WMMA tiles
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    #pragma unroll
    for (int wm = 0; wm < WMMA_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_TILES_N; wn++) {
            wmma::fill_fragment(c_frag[wm][wn], 0.0f);
        }
    }

    // Load config
    // A: BM * BK = 256 * 32 = 8192 elements, 256 threads -> 32 per thread
    // B: BK * BN = 32 * 128 = 4096 elements, 256 threads -> 16 per thread
    constexpr int A_LOADS = (BM * BK) / BLOCK_SIZE;  // 32
    constexpr int B_LOADS = (BK * BN) / BLOCK_SIZE;  // 16

    const int numKTiles = N / BK;
    int writeIdx = 0;

    // Load first tile
    #pragma unroll
    for (int i = 0; i < A_LOADS; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BK;
        int col = linearIdx % BK;

        __nv_bfloat16 val = (blockM + row < N && col < N) ?
            A[(blockM + row) * N + col] : __float2bfloat16(0.0f);
        As[writeIdx][col][row] = val;
    }

    #pragma unroll
    for (int i = 0; i < B_LOADS; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BN;
        int col = linearIdx % BN;

        __nv_bfloat16 val = (row < N && blockN + col < N) ?
            B[row * N + blockN + col] : __float2bfloat16(0.0f);
        Bs[writeIdx][row][col] = val;
    }

    __syncthreads();

    // Main loop with double buffering
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readIdx = writeIdx;
        writeIdx = 1 - writeIdx;

        // Prefetch next tile
        if (kTile + 1 < numKTiles) {
            int kOffset = (kTile + 1) * BK;

            #pragma unroll
            for (int i = 0; i < A_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BK;
                int col = linearIdx % BK;

                __nv_bfloat16 val = (blockM + row < N && kOffset + col < N) ?
                    A[(blockM + row) * N + kOffset + col] : __float2bfloat16(0.0f);
                As[writeIdx][col][row] = val;
            }

            #pragma unroll
            for (int i = 0; i < B_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BN;
                int col = linearIdx % BN;

                __nv_bfloat16 val = (kOffset + row < N && blockN + col < N) ?
                    B[(kOffset + row) * N + blockN + col] : __float2bfloat16(0.0f);
                Bs[writeIdx][row][col] = val;
            }
        }

        // Compute - 2 k-steps for BK=32
        #pragma unroll
        for (int k = 0; k < BK / WMMA_K; k++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
                a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
                b_frag[WMMA_TILES_N];

            // Load all A fragments for this warp (8 tiles)
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                int aRow = warpM * WM + wm * WMMA_M;
                int aK = k * WMMA_K;
                wmma::load_matrix_sync(a_frag[wm], &As[readIdx][aK][aRow], BM + 8);
            }

            // Load all B fragments for this warp (2 tiles)
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                int bCol = warpN * WN + wn * WMMA_N;
                int bK = k * WMMA_K;
                wmma::load_matrix_sync(b_frag[wn], &Bs[readIdx][bK][bCol], BN + 8);
            }

            // 8x2 = 16 WMMA operations per warp per k-step
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

MatmulWmmaOptBf16V5::MatmulWmmaOptBf16V5(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("BF16 WMMA requires SM 8.0+");
    }

    if (N % 256 != 0) {
        throw std::runtime_error("N must be multiple of 256 for V5");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulWmmaOptBf16V5::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16OptV5<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16OptV5<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulWmmaOptBf16V5Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWmmaOptBf16V5::~MatmulWmmaOptBf16V5() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
