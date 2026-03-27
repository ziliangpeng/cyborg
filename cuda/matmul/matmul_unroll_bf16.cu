#include "matmul_unroll_bf16.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Process 4 K tiles per main loop iteration
// ============================================================================

#define BM 128
#define BN 128
#define BK 16           // Single K tile size
#define K_UNROLL 4      // Process 4 K tiles per main loop

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

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16Unroll(const float* input, __nv_bfloat16* output, int size) {
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
// Unrolled BF16 Kernel - 4x K unrolling
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
matmulUnrollBf16Kernel(const __nv_bfloat16* __restrict__ A,
                       const __nv_bfloat16* __restrict__ B,
                       float* __restrict__ C,
                       int N) {
#if __CUDA_ARCH__ >= 800
    // Shared memory for K_UNROLL tiles (no double buffering needed with full unroll)
    __shared__ __nv_bfloat16 As[K_UNROLL][BK][BM + 8];
    __shared__ __nv_bfloat16 Bs[K_UNROLL][BK][BN + 8];

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

    const int numKUnrolls = N / (BK * K_UNROLL);

    // Main loop - process K_UNROLL tiles per iteration
    for (int kUnroll = 0; kUnroll < numKUnrolls; kUnroll++) {
        int kBase = kUnroll * BK * K_UNROLL;

        // Load all K_UNROLL tiles
        #pragma unroll
        for (int u = 0; u < K_UNROLL; u++) {
            int kOffset = kBase + u * BK;

            #pragma unroll
            for (int i = 0; i < A_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BK;
                int col = linearIdx % BK;

                __nv_bfloat16 val = A[(blockM + row) * N + kOffset + col];
                As[u][col][row] = val;
            }

            #pragma unroll
            for (int i = 0; i < B_LOADS; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BN;
                int col = linearIdx % BN;

                __nv_bfloat16 val = B[(kOffset + row) * N + blockN + col];
                Bs[u][row][col] = val;
            }
        }

        __syncthreads();

        // Compute on all K_UNROLL tiles
        #pragma unroll
        for (int u = 0; u < K_UNROLL; u++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
                a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
                b_frag[WMMA_TILES_N];

            // Load A fragments
            #pragma unroll
            for (int m = 0; m < WMMA_TILES_M; m++) {
                int aRow = warpM * WM + m * WMMA_M;
                wmma::load_matrix_sync(a_frag[m], &As[u][0][aRow], BM + 8);
            }

            // Load B fragments
            #pragma unroll
            for (int n = 0; n < WMMA_TILES_N; n++) {
                int bCol = warpN * WN + n * WMMA_N;
                wmma::load_matrix_sync(b_frag[n], &Bs[u][0][bCol], BN + 8);
            }

            // Compute with RLRL pattern
            #pragma unroll
            for (int m = 0; m < WMMA_TILES_M; m++) {
                wmma::mma_sync(c_frag[m][0], a_frag[m], b_frag[0], c_frag[m][0]);
                wmma::mma_sync(c_frag[m][1], a_frag[m], b_frag[1], c_frag[m][1]);
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

MatmulUnrollBf16::MatmulUnrollBf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("Unroll BF16 requires SM 8.0+");
    }

    if (N % 128 != 0) {
        throw std::runtime_error("N must be multiple of 128");
    }

    if (N % (BK * K_UNROLL) != 0) {
        throw std::runtime_error("N must be multiple of BK * K_UNROLL (64)");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulUnrollBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16Unroll<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16Unroll<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulUnrollBf16Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulUnrollBf16::~MatmulUnrollBf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
