#include "matmul_compact_bf16.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Compact Configuration - smallest practical tiles for max occupancy
// Targeting 4+ blocks per SM
// ============================================================================

#define BM 64
#define BN 64
#define BK 16          // Must be 16 for WMMA

#define WM 32
#define WN 32

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M (BM / WM)   // 2
#define WARPS_N (BN / WN)   // 2
#define NUM_WARPS (WARPS_M * WARPS_N)  // 4

#define WMMA_TILES_M (WM / WMMA_M)  // 2
#define WMMA_TILES_N (WN / WMMA_N)  // 2

#define BLOCK_SIZE (NUM_WARPS * 32)  // 128

// Shared memory: 2 * (16*(64+8) + 16*(64+8)) * 2 bytes = 4608 bytes
// With 64KB shared memory, we can have up to 14 blocks per SM

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16Compact(const float* input, __nv_bfloat16* output, int size) {
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
// Compact BF16 Kernel - Focus on high occupancy
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 8)  // Target high occupancy
matmulCompactBf16Kernel(const __nv_bfloat16* __restrict__ A,
                        const __nv_bfloat16* __restrict__ B,
                        float* __restrict__ C,
                        int N) {
#if __CUDA_ARCH__ >= 800
    // Double buffered shared memory
    __shared__ __nv_bfloat16 As[2][BK][BM + 8];
    __shared__ __nv_bfloat16 Bs[2][BK][BN + 8];

    const int warpId = threadIdx.x / 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Accumulators - 2x2 WMMA tiles per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    #pragma unroll
    for (int m = 0; m < WMMA_TILES_M; m++) {
        #pragma unroll
        for (int n = 0; n < WMMA_TILES_N; n++) {
            wmma::fill_fragment(c_frag[m][n], 0.0f);
        }
    }

    // Load config
    constexpr int A_LOADS = (BM * BK) / BLOCK_SIZE;  // 8
    constexpr int B_LOADS = (BK * BN) / BLOCK_SIZE;  // 8

    const int numKTiles = N / BK;

    // Load first tile
    #pragma unroll
    for (int i = 0; i < A_LOADS; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BK;
        int col = linearIdx % BK;

        __nv_bfloat16 val = (blockM + row < N && col < N) ?
            A[(blockM + row) * N + col] : __float2bfloat16(0.0f);
        As[0][col][row] = val;
    }

    #pragma unroll
    for (int i = 0; i < B_LOADS; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BN;
        int col = linearIdx % BN;

        __nv_bfloat16 val = (row < N && blockN + col < N) ?
            B[row * N + blockN + col] : __float2bfloat16(0.0f);
        Bs[0][row][col] = val;
    }

    __syncthreads();

    // Main loop with double buffering
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readBuf = kTile % 2;
        int writeBuf = 1 - readBuf;
        int kOffset = (kTile + 1) * BK;

        // Load next tile while computing current
        if (kTile + 1 < numKTiles) {
            #pragma unroll
            for (int i = 0; i < A_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BK;
                int col = linearIdx % BK;

                __nv_bfloat16 val = (blockM + row < N && kOffset + col < N) ?
                    A[(blockM + row) * N + kOffset + col] : __float2bfloat16(0.0f);
                As[writeBuf][col][row] = val;
            }

            #pragma unroll
            for (int i = 0; i < B_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BN;
                int col = linearIdx % BN;

                __nv_bfloat16 val = (kOffset + row < N && blockN + col < N) ?
                    B[(kOffset + row) * N + blockN + col] : __float2bfloat16(0.0f);
                Bs[writeBuf][row][col] = val;
            }
        }

        // Load fragments and compute
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
            a_frag[WMMA_TILES_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
            b_frag[WMMA_TILES_N];

        #pragma unroll
        for (int m = 0; m < WMMA_TILES_M; m++) {
            int aRow = warpM * WM + m * WMMA_M;
            wmma::load_matrix_sync(a_frag[m], &As[readBuf][0][aRow], BM + 8);
        }

        #pragma unroll
        for (int n = 0; n < WMMA_TILES_N; n++) {
            int bCol = warpN * WN + n * WMMA_N;
            wmma::load_matrix_sync(b_frag[n], &Bs[readBuf][0][bCol], BN + 8);
        }

        // Compute
        #pragma unroll
        for (int m = 0; m < WMMA_TILES_M; m++) {
            #pragma unroll
            for (int n = 0; n < WMMA_TILES_N; n++) {
                wmma::mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
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

MatmulCompactBf16::MatmulCompactBf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("Compact BF16 requires SM 8.0+");
    }

    if (N % 64 != 0) {
        throw std::runtime_error("N must be multiple of 64 for compact kernel");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulCompactBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16Compact<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16Compact<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulCompactBf16Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulCompactBf16::~MatmulCompactBf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
