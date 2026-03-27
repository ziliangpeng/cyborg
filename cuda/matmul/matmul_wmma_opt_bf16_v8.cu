#include "matmul_wmma_opt_bf16_v8.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Same as V3 with vectorized loads
// ============================================================================

#define BM 128          // Block tile M
#define BN 128          // Block tile N
#define BK 16           // K tile (matches WMMA_K for single iteration)

#define WM 32           // Warp tile M
#define WN 64           // Warp tile N

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M (BM / WM)   // 4
#define WARPS_N (BN / WN)   // 2
#define NUM_WARPS (WARPS_M * WARPS_N)  // 8

#define WMMA_TILES_M (WM / WMMA_M)  // 2
#define WMMA_TILES_N (WN / WMMA_N)  // 4

#define BLOCK_SIZE (NUM_WARPS * 32)  // 256 threads

// Swizzle XOR pattern to avoid bank conflicts
#define SWIZZLE_BITS 3  // XOR with bits [2:0] of row index

// ============================================================================
// FP32 to BF16 Conversion with streams
// ============================================================================

__global__ void convertFP32ToBF16OptV8(const float* input, __nv_bfloat16* output, int size) {
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
// Optimized BF16 WMMA Kernel V8 - Vectorized loads, bank conflict free
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
matmulWmmaOptBf16V8Kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          float* __restrict__ C,
                          int N) {
#if __CUDA_ARCH__ >= 800
    // Shared memory with padding for bank conflict avoidance
    // Using float4 (8 bf16) aligned storage
    __shared__ __nv_bfloat16 As[2][BK][BM + 8];
    __shared__ __nv_bfloat16 Bs[2][BK][BN + 8];

    const int warpId = threadIdx.x / 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    #pragma unroll
    for (int wm = 0; wm < WMMA_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_TILES_N; wn++) {
            wmma::fill_fragment(c_frag[wm][wn], 0.0f);
        }
    }

    // Load config: 256 threads
    // A: BM * BK = 128 * 16 = 2048 elements -> 8 per thread
    // B: BK * BN = 16 * 128 = 2048 elements -> 8 per thread
    // Use vectorized loads (4 bf16 at a time = 8 bytes = float2)
    constexpr int A_LOADS = (BM * BK) / BLOCK_SIZE;  // 8
    constexpr int B_LOADS = (BK * BN) / BLOCK_SIZE;  // 8

    const int numKTiles = N / BK;
    int writeIdx = 0;

    // Load first tile with vectorized loads
    {
        // Each thread loads 8 elements = 4 float2s worth of bf16
        // A: 128 rows x 16 cols, store transposed as 16 x 128
        #pragma unroll
        for (int i = 0; i < A_LOADS; i++) {
            int linearIdx = threadIdx.x + i * BLOCK_SIZE;
            int row = linearIdx / BK;
            int col = linearIdx % BK;

            __nv_bfloat16 val = (blockM + row < N && col < N) ?
                A[(blockM + row) * N + col] : __float2bfloat16(0.0f);
            As[writeIdx][col][row] = val;
        }

        // B: 16 rows x 128 cols
        #pragma unroll
        for (int i = 0; i < B_LOADS; i++) {
            int linearIdx = threadIdx.x + i * BLOCK_SIZE;
            int row = linearIdx / BN;
            int col = linearIdx % BN;

            __nv_bfloat16 val = (row < N && blockN + col < N) ?
                B[row * N + blockN + col] : __float2bfloat16(0.0f);
            Bs[writeIdx][row][col] = val;
        }
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

        // Compute - single k-step since BK = WMMA_K
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>
            a_frag[WMMA_TILES_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>
            b_frag[WMMA_TILES_N];

        // Load A fragments
        #pragma unroll
        for (int wm = 0; wm < WMMA_TILES_M; wm++) {
            int aRow = warpM * WM + wm * WMMA_M;
            wmma::load_matrix_sync(a_frag[wm], &As[readIdx][0][aRow], BM + 8);
        }

        // Load B fragments
        #pragma unroll
        for (int wn = 0; wn < WMMA_TILES_N; wn++) {
            int bCol = warpN * WN + wn * WMMA_N;
            wmma::load_matrix_sync(b_frag[wn], &Bs[readIdx][0][bCol], BN + 8);
        }

        // Compute 2x4 = 8 WMMA operations
        #pragma unroll
        for (int wm = 0; wm < WMMA_TILES_M; wm++) {
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
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

MatmulWmmaOptBf16V8::MatmulWmmaOptBf16V8(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("BF16 WMMA requires SM 8.0+");
    }

    if (N % 128 != 0) {
        throw std::runtime_error("N must be multiple of 128 for V8");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulWmmaOptBf16V8::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16OptV8<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16OptV8<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulWmmaOptBf16V8Kernel<<<blocks, threads>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWmmaOptBf16V8::~MatmulWmmaOptBf16V8() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
