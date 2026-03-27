#include "matmul_large_tile_bf16.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Large tiles based on successful HGEMM implementations
// ============================================================================

#define BM 256          // Block tile M - large for data reuse
#define BN 128          // Block tile N
#define BK 32           // K tile

#define WM 64           // Warp tile M
#define WN 64           // Warp tile N

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M (BM / WM)   // 4
#define WARPS_N (BN / WN)   // 2
#define NUM_WARPS (WARPS_M * WARPS_N)  // 8

// Each warp computes 4x4 WMMA tiles = 64x64 output
#define WMMA_TILES_M (WM / WMMA_M)  // 4
#define WMMA_TILES_N (WN / WMMA_N)  // 4

#define BLOCK_SIZE (NUM_WARPS * 32)  // 256 threads

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16LargeTile(const float* input, __nv_bfloat16* output, int size) {
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
// Large Tile Kernel with Register Reuse
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)  // Lower occupancy for larger tiles
matmulLargeTileBf16Kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          float* __restrict__ C,
                          int N) {
#if __CUDA_ARCH__ >= 800
    // Shared memory with padding for bank conflict avoidance
    // A: BK x BM (transposed/col-major for WMMA)
    // B: BK x BN (row-major)
    __shared__ __nv_bfloat16 As[2][BK][BM + 8];
    __shared__ __nv_bfloat16 Bs[2][BK][BN + 8];

    const int warpId = threadIdx.x / 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Large accumulator array - each warp computes 4x4 WMMA tiles = 64x64
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    #pragma unroll
    for (int m = 0; m < WMMA_TILES_M; m++) {
        #pragma unroll
        for (int n = 0; n < WMMA_TILES_N; n++) {
            wmma::fill_fragment(c_frag[m][n], 0.0f);
        }
    }

    // Load configuration
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
        int m = linearIdx % BM;
        int k = linearIdx / BM;

        int gRow = blockM + m;
        int gCol = k;

        __nv_bfloat16 val = (gRow < N && gCol < N) ?
            A[gRow * N + gCol] : __float2bfloat16(0.0f);
        As[writeIdx][k][m] = val;
    }

    #pragma unroll
    for (int i = 0; i < B_LOADS; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int k = linearIdx / BN;
        int n = linearIdx % BN;

        int gRow = k;
        int gCol = blockN + n;

        __nv_bfloat16 val = (gRow < N && gCol < N) ?
            B[gRow * N + gCol] : __float2bfloat16(0.0f);
        Bs[writeIdx][k][n] = val;
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
                int m = linearIdx % BM;
                int k = linearIdx / BM;

                int gRow = blockM + m;
                int gCol = kOffset + k;

                __nv_bfloat16 val = (gRow < N && gCol < N) ?
                    A[gRow * N + gCol] : __float2bfloat16(0.0f);
                As[writeIdx][k][m] = val;
            }

            #pragma unroll
            for (int i = 0; i < B_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int k = linearIdx / BN;
                int n = linearIdx % BN;

                int gRow = kOffset + k;
                int gCol = blockN + n;

                __nv_bfloat16 val = (gRow < N && gCol < N) ?
                    B[gRow * N + gCol] : __float2bfloat16(0.0f);
                Bs[writeIdx][k][n] = val;
            }
        }

        // Compute with register reuse pattern
        // Process 2 K steps (BK=32, WMMA_K=16)
        #pragma unroll
        for (int kStep = 0; kStep < BK / WMMA_K; kStep++) {
            // Load A fragments (4 tiles for this warp)
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
                a_frag[WMMA_TILES_M];

            #pragma unroll
            for (int m = 0; m < WMMA_TILES_M; m++) {
                int aRow = warpM * WM + m * WMMA_M;
                int aK = kStep * WMMA_K;
                wmma::load_matrix_sync(a_frag[m], &As[readIdx][aK][aRow], BM + 8);
            }

            // Load B fragments (4 tiles for this warp)
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
                b_frag[WMMA_TILES_N];

            #pragma unroll
            for (int n = 0; n < WMMA_TILES_N; n++) {
                int bCol = warpN * WN + n * WMMA_N;
                int bK = kStep * WMMA_K;
                wmma::load_matrix_sync(b_frag[n], &Bs[readIdx][bK][bCol], BN + 8);
            }

            // Compute 4x4 = 16 WMMA operations per warp
            // Use "RLRL" pattern for A register reuse
            // Process columns 0,1 then 3,2 (alternating direction)
            #pragma unroll
            for (int m = 0; m < WMMA_TILES_M; m++) {
                // Forward direction: n = 0, 1
                wmma::mma_sync(c_frag[m][0], a_frag[m], b_frag[0], c_frag[m][0]);
                wmma::mma_sync(c_frag[m][1], a_frag[m], b_frag[1], c_frag[m][1]);
                // Backward direction: n = 3, 2
                wmma::mma_sync(c_frag[m][3], a_frag[m], b_frag[3], c_frag[m][3]);
                wmma::mma_sync(c_frag[m][2], a_frag[m], b_frag[2], c_frag[m][2]);
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

MatmulLargeTileBf16::MatmulLargeTileBf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("Large tile BF16 requires SM 8.0+");
    }

    if (N % 256 != 0) {
        throw std::runtime_error("N must be multiple of 256");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulLargeTileBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16LargeTile<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16LargeTile<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulLargeTileBf16Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulLargeTileBf16::~MatmulLargeTileBf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
