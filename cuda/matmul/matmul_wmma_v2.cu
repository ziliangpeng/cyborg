#include "matmul_wmma_v2.h"
#include "cuda_utils.h"
#include <mma.h>
#include <cuda_pipeline_primitives.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Kernel Configuration - Aggressive settings for H100
// ============================================================================

// Block tile dimensions
constexpr int BM = 256;         // Block tile M (increased)
constexpr int BN = 128;         // Block tile N
constexpr int BK = 64;          // Block tile K (increased for better compute/memory ratio)

// Warp tile dimensions
constexpr int WM = 64;          // Warp tile M
constexpr int WN = 32;          // Warp tile N

// WMMA tile dimensions (hardware)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Number of warps per block
constexpr int WARPS_M = BM / WM;   // 256/64 = 4 warps in M
constexpr int WARPS_N = BN / WN;   // 128/32 = 4 warps in N
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 16 warps = 512 threads

// WMMA tiles per warp
constexpr int WMMA_TILES_M = WM / WMMA_M;  // 64/16 = 4 tiles in M
constexpr int WMMA_TILES_N = WN / WMMA_N;  // 32/16 = 2 tiles in N

// Threads per block
constexpr int BLOCK_SIZE = NUM_WARPS * 32;  // 512 threads

// Number of pipeline stages for double buffering
constexpr int NUM_STAGES = 2;

// ============================================================================
// FP32 to FP16 Conversion Kernel (vectorized)
// ============================================================================

__global__ void convertFP32ToFP16V2(const float* input, half* output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 in4 = *reinterpret_cast<const float4*>(&input[idx]);
        half2 h0 = __floats2half2_rn(in4.x, in4.y);
        half2 h1 = __floats2half2_rn(in4.z, in4.w);
        *reinterpret_cast<half2*>(&output[idx]) = h0;
        *reinterpret_cast<half2*>(&output[idx + 2]) = h1;
    }
}

// ============================================================================
// Optimized WMMA Kernel V2 - Aggressive optimizations
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE)
matmulWmmaV2Kernel(const half* __restrict__ A,
                   const half* __restrict__ B,
                   float* __restrict__ C,
                   int N) {
    // Shared memory with double buffering and padding for bank conflicts
    // As[stage][k][m] - A is loaded transposed
    // Bs[stage][k][n] - B is loaded normally
    __shared__ half As[NUM_STAGES][BK][BM + 16];
    __shared__ half Bs[NUM_STAGES][BK][BN + 16];

    // Thread/warp indices
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    // Block position
    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Declare accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WMMA_TILES_M][WMMA_TILES_N];

    // Initialize accumulators
    #pragma unroll
    for (int wm = 0; wm < WMMA_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WMMA_TILES_N; wn++) {
            wmma::fill_fragment(c_frag[wm][wn], 0.0f);
        }
    }

    // ========================================================================
    // Memory loading configuration
    // A tile: BM x BK = 256 x 64 = 16384 half elements
    // B tile: BK x BN = 64 x 128 = 8192 half elements
    // Total: 24576 elements, 512 threads -> 48 elements per thread
    // Use 8 halfs (16 bytes = 128 bits) per load where possible
    // ========================================================================

    const int numKTiles = N / BK;

    // Load first tile into stage 0
    int stage = 0;

    // Load A: 256x64 = 16384 halfs, 512 threads, 32 elements per thread
    #pragma unroll
    for (int i = 0; i < (BM * BK) / BLOCK_SIZE; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BK;
        int col = linearIdx % BK;

        half val = (blockM + row < N && col < N) ? A[(blockM + row) * N + col] : __float2half(0.0f);
        As[stage][col][row] = val;  // Store transposed
    }

    // Load B: 64x128 = 8192 halfs, 512 threads, 16 elements per thread
    #pragma unroll
    for (int i = 0; i < (BK * BN) / BLOCK_SIZE; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int row = linearIdx / BN;
        int col = linearIdx % BN;

        half val = (row < N && blockN + col < N) ? B[row * N + blockN + col] : __float2half(0.0f);
        Bs[stage][row][col] = val;
    }

    __syncthreads();

    // ========================================================================
    // Main K-loop with double buffering
    // ========================================================================

    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readStage = stage;
        int writeStage = 1 - stage;
        stage = writeStage;

        // Start loading next tile asynchronously (if not last)
        if (kTile + 1 < numKTiles) {
            int kOffset = (kTile + 1) * BK;

            #pragma unroll
            for (int i = 0; i < (BM * BK) / BLOCK_SIZE; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BK;
                int col = linearIdx % BK;

                half val = (blockM + row < N && kOffset + col < N) ?
                           A[(blockM + row) * N + kOffset + col] : __float2half(0.0f);
                As[writeStage][col][row] = val;
            }

            #pragma unroll
            for (int i = 0; i < (BK * BN) / BLOCK_SIZE; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int row = linearIdx / BN;
                int col = linearIdx % BN;

                half val = (kOffset + row < N && blockN + col < N) ?
                           B[(kOffset + row) * N + blockN + col] : __float2half(0.0f);
                Bs[writeStage][row][col] = val;
            }
        }

        // Compute on current tile
        #pragma unroll
        for (int kWmma = 0; kWmma < BK; kWmma += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
                a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                b_frag[WMMA_TILES_N];

            // Load A fragments
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                int aRow = warpM * WM + wm * WMMA_M;
                wmma::load_matrix_sync(a_frag[wm], &As[readStage][kWmma][aRow], BM + 16);
            }

            // Load B fragments
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                int bCol = warpN * WN + wn * WMMA_N;
                wmma::load_matrix_sync(b_frag[wn], &Bs[readStage][kWmma][bCol], BN + 16);
            }

            // MMA
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
    // Store results
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

MatmulWmmaV2::MatmulWmmaV2(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 7) {
        throw std::runtime_error("WMMA requires compute capability 7.0+");
    }

    if (N % 256 != 0) {
        throw std::runtime_error("Matrix dimension N must be a multiple of 256");
    }

    cudaCheckError(cudaMalloc(&d_A_fp16, N * N * sizeof(half)));
    cudaCheckError(cudaMalloc(&d_B_fp16, N * N * sizeof(half)));
}

void MatmulWmmaV2::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToFP16V2<<<convBlocks, convThreads>>>(d_A, d_A_fp16, N * N);
    convertFP32ToFP16V2<<<convBlocks, convThreads>>>(d_B, d_B_fp16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulWmmaV2Kernel<<<blocks, threads>>>(d_A_fp16, d_B_fp16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWmmaV2::~MatmulWmmaV2() {
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
}
