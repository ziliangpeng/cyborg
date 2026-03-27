#include "matmul_wmma_optimized.h"
#include "cuda_utils.h"
#include <mma.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Kernel Configuration
// ============================================================================

// Block tile dimensions (output tile computed by one thread block)
#define BM 128          // Block tile M
#define BN 256          // Block tile N
#define BK 32           // Block tile K (depth of one iteration)

// Warp tile dimensions (output tile computed by one warp)
#define WM 64           // Warp tile M
#define WN 64           // Warp tile N

// WMMA tile dimensions (hardware instruction size)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Number of warps per block
#define WARPS_M (BM / WM)   // 128/64 = 2 warps in M
#define WARPS_N (BN / WN)   // 256/64 = 4 warps in N
#define NUM_WARPS (WARPS_M * WARPS_N)  // 8 warps = 256 threads

// WMMA tiles per warp
#define WMMA_TILES_M (WM / WMMA_M)  // 64/16 = 4 tiles in M
#define WMMA_TILES_N (WN / WMMA_N)  // 64/16 = 4 tiles in N

// Threads per block
#define BLOCK_SIZE (NUM_WARPS * 32)  // 256 threads

// ============================================================================
// FP32 to FP16 Conversion Kernel
// ============================================================================

__global__ void convertFP32ToFP16Optimized(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Vectorized conversion: 4 elements at a time
    if (idx * 4 + 3 < size) {
        float4 in4 = *reinterpret_cast<const float4*>(&input[idx * 4]);
        half2 h0 = __floats2half2_rn(in4.x, in4.y);
        half2 h1 = __floats2half2_rn(in4.z, in4.w);
        *reinterpret_cast<half2*>(&output[idx * 4]) = h0;
        *reinterpret_cast<half2*>(&output[idx * 4 + 2]) = h1;
    } else if (idx * 4 < size) {
        // Handle remainder
        for (int i = 0; i < 4 && idx * 4 + i < size; i++) {
            output[idx * 4 + i] = __float2half(input[idx * 4 + i]);
        }
    }
}

// ============================================================================
// Optimized WMMA Kernel with Double Buffering
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE)
matmulWmmaOptimizedKernel(const half* __restrict__ A,
                          const half* __restrict__ B,
                          float* __restrict__ C,
                          int N) {
    // Shared memory with double buffering
    // Add padding to avoid bank conflicts (16 half = 32 bytes = 1 bank width)
    __shared__ half As[2][BK][BM + 8];  // Transposed: As[k][m]
    __shared__ half Bs[2][BK][BN + 8];  // Bs[k][n]

    // Warp and lane indices
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Warp position in the block tile
    const int warpM = warpId / WARPS_N;  // 0-1
    const int warpN = warpId % WARPS_N;  // 0-3

    // Block position in output
    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Declare WMMA fragments for accumulator
    // Each warp computes 4x4 = 16 WMMA tiles
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

    // ========================================================================
    // Collaborative loading setup
    // 256 threads load BM*BK = 128*32 = 4096 half elements for A
    // 256 threads load BK*BN = 32*256 = 8192 half elements for B
    // ========================================================================

    // Number of K iterations
    const int numKTiles = N / BK;

    // ========================================================================
    // Load first tile (no double buffer swap on first iteration)
    // ========================================================================
    int writeIdx = 0;

    // Load A tile - each thread loads multiple elements
    // A: (BM x BK) = 128 x 32 = 4096 half elements
    // 256 threads, each thread loads 4096/256 = 16 elements
    #pragma unroll
    for (int i = 0; i < (BM * BK) / BLOCK_SIZE; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BK;  // row in A tile [0, BM)
        int col = linearIdx % BK;  // col in A tile [0, BK)

        half val = __float2half(0.0f);
        if (blockM + row < N && col < N) {
            val = A[(blockM + row) * N + col];
        }
        // Store transposed: As[k][m]
        As[writeIdx][col][row] = val;
    }

    // Load B tile
    // B: (BK x BN) = 32 x 256 = 8192 half elements
    // 256 threads, each thread loads 8192/256 = 32 elements
    #pragma unroll
    for (int i = 0; i < (BK * BN) / BLOCK_SIZE; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BN;  // row in B tile [0, BK)
        int col = linearIdx % BN;  // col in B tile [0, BN)

        half val = __float2half(0.0f);
        if (row < N && blockN + col < N) {
            val = B[row * N + blockN + col];
        }
        Bs[writeIdx][row][col] = val;
    }

    __syncthreads();

    // ========================================================================
    // Main K-loop with double buffering
    // ========================================================================

    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readIdx = writeIdx;
        writeIdx = 1 - writeIdx;

        // Prefetch next tile (if not last iteration)
        if (kTile + 1 < numKTiles) {
            int kOffset = (kTile + 1) * BK;

            // Load A tile for next iteration
            #pragma unroll
            for (int i = 0; i < (BM * BK) / BLOCK_SIZE; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BK;
                int col = linearIdx % BK;

                half val = __float2half(0.0f);
                if (blockM + row < N && kOffset + col < N) {
                    val = A[(blockM + row) * N + kOffset + col];
                }
                As[writeIdx][col][row] = val;
            }

            // Load B tile for next iteration
            #pragma unroll
            for (int i = 0; i < (BK * BN) / BLOCK_SIZE; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BN;
                int col = linearIdx % BN;

                half val = __float2half(0.0f);
                if (kOffset + row < N && blockN + col < N) {
                    val = B[(kOffset + row) * N + blockN + col];
                }
                Bs[writeIdx][row][col] = val;
            }
        }

        // ====================================================================
        // Compute on current tile (read from readIdx buffer)
        // ====================================================================

        // Loop over BK in chunks of WMMA_K
        #pragma unroll
        for (int kWmma = 0; kWmma < BK; kWmma += WMMA_K) {
            // Declare fragments for A and B
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
                a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                b_frag[WMMA_TILES_N];

            // Load A fragments (col-major because As is transposed: As[k][m])
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                int aRow = warpM * WM + wm * WMMA_M;
                wmma::load_matrix_sync(a_frag[wm],
                    &As[readIdx][kWmma][aRow],
                    BM + 8);  // stride is BM + padding
            }

            // Load B fragments (row-major: Bs[k][n])
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                int bCol = warpN * WN + wn * WMMA_N;
                wmma::load_matrix_sync(b_frag[wn],
                    &Bs[readIdx][kWmma][bCol],
                    BN + 8);  // stride is BN + padding
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

    // ========================================================================
    // Store results to global memory
    // ========================================================================

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
}

// ============================================================================
// Class Implementation
// ============================================================================

MatmulWmmaOptimized::MatmulWmmaOptimized(int N, int blockDim) : N(N) {
    // Check compute capability
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 7) {
        throw std::runtime_error("WMMA requires compute capability 7.0+ (Volta or newer)");
    }

    // Check matrix dimension compatibility
    if (N % BM != 0 || N % BN != 0) {
        throw std::runtime_error("Matrix dimension N must be a multiple of block tile sizes (128, 256)");
    }

    // Allocate FP16 buffers
    cudaCheckError(cudaMalloc(&d_A_fp16, N * N * sizeof(half)));
    cudaCheckError(cudaMalloc(&d_B_fp16, N * N * sizeof(half)));
}

void MatmulWmmaOptimized::execute(const float *d_A, const float *d_B, float *d_C) {
    // Convert FP32 inputs to FP16 (vectorized)
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToFP16Optimized<<<convBlocks, convThreads>>>(d_A, d_A_fp16, N * N);
    convertFP32ToFP16Optimized<<<convBlocks, convThreads>>>(d_B, d_B_fp16, N * N);

    // Launch optimized WMMA kernel
    dim3 blockDim(BLOCK_SIZE);  // 256 threads
    dim3 gridDim((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulWmmaOptimizedKernel<<<gridDim, blockDim>>>(d_A_fp16, d_B_fp16, d_C, N);

    cudaCheckError(cudaGetLastError());
}

MatmulWmmaOptimized::~MatmulWmmaOptimized() {
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
}
