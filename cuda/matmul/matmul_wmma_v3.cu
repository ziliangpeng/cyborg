#include "matmul_wmma_v3.h"
#include "cuda_utils.h"
#include <mma.h>
#include <iostream>
#include <stdexcept>

using namespace nvcuda;

// ============================================================================
// Configuration - Tuned for H100
// ============================================================================

// Block tiles - same as V1 but with larger K
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;  // Increased from 32 to 64 for better compute/memory ratio

// Warp tiles
constexpr int WM = 64;
constexpr int WN = 32;

// WMMA hardware tiles
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warp organization
constexpr int WARPS_M = BM / WM;  // 2
constexpr int WARPS_N = BN / WN;  // 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 8

// WMMA tiles per warp
constexpr int WMMA_TILES_M = WM / WMMA_M;  // 4
constexpr int WMMA_TILES_N = WN / WMMA_N;  // 2

constexpr int BLOCK_SIZE = NUM_WARPS * 32;  // 256 threads

// Padding to avoid bank conflicts
constexpr int A_PAD = 8;
constexpr int B_PAD = 8;

// ============================================================================
// FP32 to FP16 Conversion
// ============================================================================

__global__ void convertFP32ToFP16V3(const float* __restrict__ input,
                                     half* __restrict__ output, int size) {
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
// WMMA Kernel V3
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE)
matmulWmmaV3Kernel(const half* __restrict__ A,
                   const half* __restrict__ B,
                   float* __restrict__ C,
                   int N) {
    // Static shared memory allocation with padding
    __shared__ half As[2][BK][BM + A_PAD];  // Transposed A: [stage][k][m]
    __shared__ half Bs[2][BK][BN + B_PAD];  // B: [stage][k][n]

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

    // Number of elements each thread loads
    // A: BM * BK = 128 * 64 = 8192, 256 threads -> 32 elements/thread
    // B: BK * BN = 64 * 128 = 8192, 256 threads -> 32 elements/thread
    constexpr int A_LOADS_PER_THREAD = (BM * BK) / BLOCK_SIZE;
    constexpr int B_LOADS_PER_THREAD = (BK * BN) / BLOCK_SIZE;

    const int numKTiles = N / BK;

    // Double buffer index
    int stage = 0;

    // Load first tile
    #pragma unroll
    for (int i = 0; i < A_LOADS_PER_THREAD; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int m = linearIdx % BM;
        int k = linearIdx / BM;

        half val = (blockM + m < N && k < N) ? A[(blockM + m) * N + k] : __float2half(0.0f);
        As[stage][k][m] = val;
    }

    #pragma unroll
    for (int i = 0; i < B_LOADS_PER_THREAD; i++) {
        int linearIdx = threadIdx.x + i * BLOCK_SIZE;
        int n = linearIdx % BN;
        int k = linearIdx / BN;

        half val = (k < N && blockN + n < N) ? B[k * N + blockN + n] : __float2half(0.0f);
        Bs[stage][k][n] = val;
    }

    __syncthreads();

    // Main loop
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readStage = stage;
        int writeStage = 1 - stage;
        stage = writeStage;

        // Prefetch next tile
        if (kTile + 1 < numKTiles) {
            int kOffset = (kTile + 1) * BK;

            #pragma unroll
            for (int i = 0; i < A_LOADS_PER_THREAD; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int m = linearIdx % BM;
                int k = linearIdx / BM;

                half val = (blockM + m < N && kOffset + k < N) ?
                           A[(blockM + m) * N + kOffset + k] : __float2half(0.0f);
                As[writeStage][k][m] = val;
            }

            #pragma unroll
            for (int i = 0; i < B_LOADS_PER_THREAD; i++) {
                int linearIdx = threadIdx.x + i * BLOCK_SIZE;
                int n = linearIdx % BN;
                int k = linearIdx / BN;

                half val = (kOffset + k < N && blockN + n < N) ?
                           B[(kOffset + k) * N + blockN + n] : __float2half(0.0f);
                Bs[writeStage][k][n] = val;
            }
        }

        // Compute on current tile
        #pragma unroll
        for (int kWmma = 0; kWmma < BK; kWmma += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
                a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                b_frag[WMMA_TILES_N];

            // Load A fragments (col-major, As[k][m])
            #pragma unroll
            for (int wm = 0; wm < WMMA_TILES_M; wm++) {
                int aRow = warpM * WM + wm * WMMA_M;
                wmma::load_matrix_sync(a_frag[wm],
                    &As[readStage][kWmma][aRow], BM + A_PAD);
            }

            // Load B fragments (row-major, Bs[k][n])
            #pragma unroll
            for (int wn = 0; wn < WMMA_TILES_N; wn++) {
                int bCol = warpN * WN + wn * WMMA_N;
                wmma::load_matrix_sync(b_frag[wn],
                    &Bs[readStage][kWmma][bCol], BN + B_PAD);
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
}

// ============================================================================
// Class Implementation
// ============================================================================

MatmulWmmaV3::MatmulWmmaV3(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 7) {
        throw std::runtime_error("WMMA requires SM 7.0+");
    }

    if (N % 128 != 0) {
        throw std::runtime_error("N must be multiple of 128");
    }

    cudaCheckError(cudaMalloc(&d_A_fp16, N * N * sizeof(half)));
    cudaCheckError(cudaMalloc(&d_B_fp16, N * N * sizeof(half)));
}

void MatmulWmmaV3::execute(const float *d_A, const float *d_B, float *d_C) {
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToFP16V3<<<convBlocks, convThreads>>>(d_A, d_A_fp16, N * N);
    convertFP32ToFP16V3<<<convBlocks, convThreads>>>(d_B, d_B_fp16, N * N);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmulWmmaV3Kernel<<<blocks, threads>>>(d_A_fp16, d_B_fp16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWmmaV3::~MatmulWmmaV3() {
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
}
