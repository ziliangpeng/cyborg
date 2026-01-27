#include "matmul_wmma_opt_bf16.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Kernel Configuration (same as wmma_opt)
// ============================================================================

#define BM 128          // Block tile M
#define BN 256          // Block tile N
#define BK 32           // Block tile K

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

// ============================================================================
// FP32 to BF16 Conversion Kernel (vectorized)
// ============================================================================

__global__ void convertFP32ToBF16Optimized(const float* input, __nv_bfloat16* output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 in4 = *reinterpret_cast<const float4*>(&input[idx]);
        __nv_bfloat162 h0 = __floats2bfloat162_rn(in4.x, in4.y);
        __nv_bfloat162 h1 = __floats2bfloat162_rn(in4.z, in4.w);
        *reinterpret_cast<__nv_bfloat162*>(&output[idx]) = h0;
        *reinterpret_cast<__nv_bfloat162*>(&output[idx + 2]) = h1;
    } else if (idx < size) {
        for (int i = 0; i < 4 && idx + i < size; i++) {
            output[idx + i] = __float2bfloat16(input[idx + i]);
        }
    }
}

// ============================================================================
// Optimized BF16 WMMA Kernel with Double Buffering
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE)
matmulWmmaOptBf16Kernel(const __nv_bfloat16* __restrict__ A,
                        const __nv_bfloat16* __restrict__ B,
                        float* __restrict__ C,
                        int N) {
#if __CUDA_ARCH__ >= 800
    // Shared memory with double buffering and padding
    __shared__ __nv_bfloat16 As[2][BK][BM + 8];  // Transposed: As[k][m]
    __shared__ __nv_bfloat16 Bs[2][BK][BN + 8];  // Bs[k][n]

    const int warpId = threadIdx.x / 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Declare WMMA fragments for accumulator
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    // Initialize accumulators to zero
    #pragma unroll
    for (int wm = 0; wm < WMMA_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_TILES_N; wn++) {
            wmma::fill_fragment(c_frag[wm][wn], 0.0f);
        }
    }

    const int numKTiles = N / BK;
    int writeIdx = 0;

    // Load first tile
    #pragma unroll
    for (int i = 0; i < (BM * BK) / BLOCK_SIZE; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BK;
        int col = linearIdx % BK;

        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (blockM + row < N && col < N) {
            val = A[(blockM + row) * N + col];
        }
        As[writeIdx][col][row] = val;
    }

    #pragma unroll
    for (int i = 0; i < (BK * BN) / BLOCK_SIZE; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BN;
        int col = linearIdx % BN;

        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (row < N && blockN + col < N) {
            val = B[row * N + blockN + col];
        }
        Bs[writeIdx][row][col] = val;
    }

    __syncthreads();

    // Main K-loop with double buffering
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readIdx = writeIdx;
        writeIdx = 1 - writeIdx;

        // Prefetch next tile
        if (kTile + 1 < numKTiles) {
            int kOffset = (kTile + 1) * BK;

            #pragma unroll
            for (int i = 0; i < (BM * BK) / BLOCK_SIZE; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BK;
                int col = linearIdx % BK;

                __nv_bfloat16 val = __float2bfloat16(0.0f);
                if (blockM + row < N && kOffset + col < N) {
                    val = A[(blockM + row) * N + kOffset + col];
                }
                As[writeIdx][col][row] = val;
            }

            #pragma unroll
            for (int i = 0; i < (BK * BN) / BLOCK_SIZE; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BN;
                int col = linearIdx % BN;

                __nv_bfloat16 val = __float2bfloat16(0.0f);
                if (kOffset + row < N && blockN + col < N) {
                    val = B[(kOffset + row) * N + blockN + col];
                }
                Bs[writeIdx][row][col] = val;
            }
        }

        // Compute on current tile
        #pragma unroll
        for (int kWmma = 0; kWmma < BK; kWmma += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
                a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
                b_frag[WMMA_TILES_N];

            // Load A fragments
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                int aRow = warpM * WM + wm * WMMA_M;
                wmma::load_matrix_sync(a_frag[wm],
                    &As[readIdx][kWmma][aRow], BM + 8);
            }

            // Load B fragments
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                int bCol = warpN * WN + wn * WMMA_N;
                wmma::load_matrix_sync(b_frag[wn],
                    &Bs[readIdx][kWmma][bCol], BN + 8);
            }

            // Matrix multiply-accumulate
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

MatmulWmmaOptBf16::MatmulWmmaOptBf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("BF16 WMMA requires compute capability 8.0+ (Ampere or newer)");
    }

    if (N % BN != 0) {
        throw std::runtime_error("Matrix dimension N must be a multiple of 256");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulWmmaOptBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16Optimized<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16Optimized<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulWmmaOptBf16Kernel<<<gridDim, blockDim>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWmmaOptBf16::~MatmulWmmaOptBf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
