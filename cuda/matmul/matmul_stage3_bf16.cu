#include "matmul_stage3_bf16.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Based on V3 (our best) with 3-stage pipelining
// ============================================================================

#define BM 128
#define BN 128
#define BK 16           // Keep small for better occupancy

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

#define BLOCK_SIZE (NUM_WARPS * 32)  // 256

#define NUM_STAGES 3

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16Stage3(const float* input, __nv_bfloat16* output, int size) {
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
// 3-Stage Pipeline Kernel
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
matmulStage3Bf16Kernel(const __nv_bfloat16* __restrict__ A,
                       const __nv_bfloat16* __restrict__ B,
                       float* __restrict__ C,
                       int N) {
#if __CUDA_ARCH__ >= 800
    // 3-stage shared memory buffers
    __shared__ __nv_bfloat16 As[NUM_STAGES][BK][BM + 8];
    __shared__ __nv_bfloat16 Bs[NUM_STAGES][BK][BN + 8];

    const int warpId = threadIdx.x / 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Accumulators
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

    // Helper lambda for loading a stage
    auto loadStage = [&](int stage, int kOffset) {
        #pragma unroll
        for (int i = 0; i < A_LOADS; i++) {
            int linearIdx = threadIdx.x + i * BLOCK_SIZE;
            int row = linearIdx / BK;
            int col = linearIdx % BK;

            __nv_bfloat16 val = (blockM + row < N && kOffset + col < N) ?
                A[(blockM + row) * N + kOffset + col] : __float2bfloat16(0.0f);
            As[stage][col][row] = val;
        }

        #pragma unroll
        for (int i = 0; i < B_LOADS; i++) {
            int linearIdx = threadIdx.x + i * BLOCK_SIZE;
            int row = linearIdx / BN;
            int col = linearIdx % BN;

            __nv_bfloat16 val = (kOffset + row < N && blockN + col < N) ?
                B[(kOffset + row) * N + blockN + col] : __float2bfloat16(0.0f);
            Bs[stage][row][col] = val;
        }
    };

    // Prologue: load first NUM_STAGES-1 tiles
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) {
        if (s < numKTiles) {
            loadStage(s, s * BK);
        }
    }

    __syncthreads();

    // Main loop
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readStage = kTile % NUM_STAGES;
        int writeStage = (kTile + NUM_STAGES - 1) % NUM_STAGES;

        // Start loading next stage (overlapped with compute)
        if (kTile + NUM_STAGES - 1 < numKTiles) {
            loadStage(writeStage, (kTile + NUM_STAGES - 1) * BK);
        }

        // Compute on current stage
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
            a_frag[WMMA_TILES_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
            b_frag[WMMA_TILES_N];

        // Load fragments
        #pragma unroll
        for (int m = 0; m < WMMA_TILES_M; m++) {
            int aRow = warpM * WM + m * WMMA_M;
            wmma::load_matrix_sync(a_frag[m], &As[readStage][0][aRow], BM + 8);
        }

        #pragma unroll
        for (int n = 0; n < WMMA_TILES_N; n++) {
            int bCol = warpN * WN + n * WMMA_N;
            wmma::load_matrix_sync(b_frag[n], &Bs[readStage][0][bCol], BN + 8);
        }

        // Compute with register reuse pattern (RLRL)
        #pragma unroll
        for (int m = 0; m < WMMA_TILES_M; m++) {
            wmma::mma_sync(c_frag[m][0], a_frag[m], b_frag[0], c_frag[m][0]);
            wmma::mma_sync(c_frag[m][1], a_frag[m], b_frag[1], c_frag[m][1]);
            wmma::mma_sync(c_frag[m][3], a_frag[m], b_frag[3], c_frag[m][3]);
            wmma::mma_sync(c_frag[m][2], a_frag[m], b_frag[2], c_frag[m][2]);
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

MatmulStage3Bf16::MatmulStage3Bf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("Stage3 BF16 requires SM 8.0+");
    }

    if (N % 128 != 0) {
        throw std::runtime_error("N must be multiple of 128");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulStage3Bf16::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16Stage3<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16Stage3<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulStage3Bf16Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulStage3Bf16::~MatmulStage3Bf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
