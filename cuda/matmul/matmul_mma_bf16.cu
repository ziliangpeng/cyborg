#include "matmul_mma_bf16.h"
#include "cuda_utils.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <stdexcept>

// ============================================================================
// PTX Helper Macros for MMA and LDMATRIX
// ============================================================================

// ldmatrix loads 8x8 matrix fragments from shared memory
// Each thread in a warp provides an address, collectively loading 4 8x8 tiles
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) \
        : "r"(addr) \
    )

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1) \
        : "r"(addr) \
    )

// MMA instruction: D = A * B + C
// m16n8k16 for BF16: A is 16x16, B is 16x8, C/D is 16x8
// Each thread holds: A[4 regs], B[2 regs], C/D[4 regs]
#define MMA_M16N8K16_BF16(D0, D1, D2, D3, A0, A1, A2, A3, B0, B1, C0, C1, C2, C3) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
        : "=f"(D0), "=f"(D1), "=f"(D2), "=f"(D3) \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), \
          "r"(B0), "r"(B1), \
          "f"(C0), "f"(C1), "f"(C2), "f"(C3) \
    )

// ============================================================================
// Configuration
// ============================================================================

// Block tile sizes
#define BM 128
#define BN 128
#define BK 32

// Warp tile sizes - each warp computes 64x32 output
#define WM 64
#define WN 32

// MMA tile sizes (m16n8k16)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// Warp configuration
#define WARPS_M (BM / WM)  // 2
#define WARPS_N (BN / WN)  // 4
#define NUM_WARPS (WARPS_M * WARPS_N)  // 8

// MMA tiles per warp
#define MMA_TILES_M (WM / MMA_M)  // 4
#define MMA_TILES_N (WN / MMA_N)  // 4

#define BLOCK_SIZE (NUM_WARPS * 32)  // 256 threads

// Pipeline stages
#define NUM_STAGES 3

// Swizzle pattern for bank conflict avoidance
// XOR row with (col >> 3) to spread across banks
#define SWIZZLE(row, col) ((row) ^ ((col) >> 3))

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16Mma(const float* input, __nv_bfloat16* output, int size) {
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
// Main MMA Kernel with cp.async and swizzled SMEM
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
matmulMmaBf16Kernel(const __nv_bfloat16* __restrict__ A,
                    const __nv_bfloat16* __restrict__ B,
                    float* __restrict__ C,
                    int N) {
#if __CUDA_ARCH__ >= 800
    // Shared memory with swizzled layout
    // A: BM x BK stored column-major for ldmatrix (BK x BM with swizzle)
    // B: BK x BN stored row-major for ldmatrix
    extern __shared__ __nv_bfloat16 smem[];

    // Layout: [stage][A or B][rows][cols]
    // A: NUM_STAGES * BK * BM
    // B: NUM_STAGES * BK * BN
    __nv_bfloat16* As = smem;
    __nv_bfloat16* Bs = smem + NUM_STAGES * BK * BM;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    // Accumulators - each thread holds partial results
    // For m16n8k16: 4 floats per MMA tile, 4x4 tiles per warp = 64 floats
    float acc[MMA_TILES_M][MMA_TILES_N][4];

    #pragma unroll
    for (int m = 0; m < MMA_TILES_M; m++) {
        #pragma unroll
        for (int n = 0; n < MMA_TILES_N; n++) {
            acc[m][n][0] = 0.0f;
            acc[m][n][1] = 0.0f;
            acc[m][n][2] = 0.0f;
            acc[m][n][3] = 0.0f;
        }
    }

    // Thread's load assignment
    // A: BM * BK = 128 * 32 = 4096 elements, 256 threads = 16 elements/thread
    // B: BK * BN = 32 * 128 = 4096 elements, 256 threads = 16 elements/thread
    const int loadAPerThread = (BM * BK) / BLOCK_SIZE;  // 16
    const int loadBPerThread = (BK * BN) / BLOCK_SIZE;  // 16

    const int numKTiles = N / BK;

    // Prologue: fill the pipeline
    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES - 1 && stage < numKTiles; stage++) {
        int kOffset = stage * BK;

        // Load A with swizzled layout (store transposed: k-major)
        #pragma unroll
        for (int i = 0; i < loadAPerThread; i++) {
            int linearIdx = threadIdx.x * loadAPerThread + i;
            int row = linearIdx % BM;  // M dimension
            int col = linearIdx / BM;  // K dimension

            int gRow = blockM + row;
            int gCol = kOffset + col;

            __nv_bfloat16 val = (gRow < N && gCol < N) ?
                A[gRow * N + gCol] : __float2bfloat16(0.0f);

            // Swizzled store: As[stage][col][row ^ (col >> 2)]
            int swizzledRow = row ^ ((col & 0x1C) >> 2);  // XOR with bits [4:2] of col
            As[stage * BK * BM + col * BM + swizzledRow] = val;
        }

        // Load B with swizzled layout (row-major)
        #pragma unroll
        for (int i = 0; i < loadBPerThread; i++) {
            int linearIdx = threadIdx.x * loadBPerThread + i;
            int row = linearIdx / BN;  // K dimension
            int col = linearIdx % BN;  // N dimension

            int gRow = kOffset + row;
            int gCol = blockN + col;

            __nv_bfloat16 val = (gRow < N && gCol < N) ?
                B[gRow * N + gCol] : __float2bfloat16(0.0f);

            // Swizzled store
            int swizzledCol = col ^ ((row & 0x1C) >> 2);
            Bs[stage * BK * BN + row * BN + swizzledCol] = val;
        }

        __pipeline_commit();
    }

    // Main loop
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int readStage = kTile % NUM_STAGES;
        int writeStage = (kTile + NUM_STAGES - 1) % NUM_STAGES;

        // Wait for current stage
        __pipeline_wait_prior(NUM_STAGES - 2);
        __syncthreads();

        // Prefetch next stage
        if (kTile + NUM_STAGES - 1 < numKTiles) {
            int kOffset = (kTile + NUM_STAGES - 1) * BK;

            #pragma unroll
            for (int i = 0; i < loadAPerThread; i++) {
                int linearIdx = threadIdx.x * loadAPerThread + i;
                int row = linearIdx % BM;
                int col = linearIdx / BM;

                int gRow = blockM + row;
                int gCol = kOffset + col;

                __nv_bfloat16 val = (gRow < N && gCol < N) ?
                    A[gRow * N + gCol] : __float2bfloat16(0.0f);

                int swizzledRow = row ^ ((col & 0x1C) >> 2);
                As[writeStage * BK * BM + col * BM + swizzledRow] = val;
            }

            #pragma unroll
            for (int i = 0; i < loadBPerThread; i++) {
                int linearIdx = threadIdx.x * loadBPerThread + i;
                int row = linearIdx / BN;
                int col = linearIdx % BN;

                int gRow = kOffset + row;
                int gCol = blockN + col;

                __nv_bfloat16 val = (gRow < N && gCol < N) ?
                    B[gRow * N + gCol] : __float2bfloat16(0.0f);

                int swizzledCol = col ^ ((row & 0x1C) >> 2);
                Bs[writeStage * BK * BN + row * BN + swizzledCol] = val;
            }

            __pipeline_commit();
        }

        // Compute using MMA instructions
        // Each warp processes its 64x32 tile using 4x4 m16n8k16 MMAs
        __nv_bfloat16* aSmem = As + readStage * BK * BM;
        __nv_bfloat16* bSmem = Bs + readStage * BK * BN;

        // Process BK in chunks of MMA_K (16)
        #pragma unroll
        for (int k = 0; k < BK / MMA_K; k++) {
            // Register fragments for this k-step
            uint32_t aFrag[MMA_TILES_M][4];  // 4 regs per m16n8k16 A operand
            uint32_t bFrag[MMA_TILES_N][2];  // 2 regs per m16n8k16 B operand

            // Load A fragments using ldmatrix pattern
            // Each warp loads from its M region
            #pragma unroll
            for (int m = 0; m < MMA_TILES_M; m++) {
                int aRow = warpM * WM + m * MMA_M + (laneId % 16);
                int aCol = k * MMA_K + (laneId / 16) * 8;

                // Apply inverse swizzle for load
                int swizzledRow = aRow ^ ((aCol & 0x1C) >> 2);
                uint32_t aAddr = __cvta_generic_to_shared(&aSmem[aCol * BM + swizzledRow]);

                LDMATRIX_X4(aFrag[m][0], aFrag[m][1], aFrag[m][2], aFrag[m][3], aAddr);
            }

            // Load B fragments
            #pragma unroll
            for (int n = 0; n < MMA_TILES_N; n++) {
                int bRow = k * MMA_K + (laneId % 16);
                int bCol = warpN * WN + n * MMA_N + (laneId / 16) * 4;

                int swizzledCol = bCol ^ ((bRow & 0x1C) >> 2);
                uint32_t bAddr = __cvta_generic_to_shared(&bSmem[bRow * BN + swizzledCol]);

                LDMATRIX_X2(bFrag[n][0], bFrag[n][1], bAddr);
            }

            // Execute MMA operations
            #pragma unroll
            for (int m = 0; m < MMA_TILES_M; m++) {
                #pragma unroll
                for (int n = 0; n < MMA_TILES_N; n++) {
                    MMA_M16N8K16_BF16(
                        acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                        aFrag[m][0], aFrag[m][1], aFrag[m][2], aFrag[m][3],
                        bFrag[n][0], bFrag[n][1],
                        acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3]
                    );
                }
            }
        }

        __syncthreads();
    }

    // Store results
    // m16n8k16 output layout: each thread holds 2x2 elements at specific positions
    #pragma unroll
    for (int m = 0; m < MMA_TILES_M; m++) {
        #pragma unroll
        for (int n = 0; n < MMA_TILES_N; n++) {
            int cRow = blockM + warpM * WM + m * MMA_M;
            int cCol = blockN + warpN * WN + n * MMA_N;

            // Thread's position in the 16x8 output tile
            int tRow = (laneId % 4) * 2;
            int tCol = (laneId / 4);

            // acc[0,1] are rows tRow, acc[2,3] are rows tRow+8
            if (cRow + tRow < N && cCol + tCol < N) {
                C[(cRow + tRow) * N + cCol + tCol] = acc[m][n][0];
            }
            if (cRow + tRow + 1 < N && cCol + tCol < N) {
                C[(cRow + tRow + 1) * N + cCol + tCol] = acc[m][n][1];
            }
            if (cRow + tRow + 8 < N && cCol + tCol < N) {
                C[(cRow + tRow + 8) * N + cCol + tCol] = acc[m][n][2];
            }
            if (cRow + tRow + 9 < N && cCol + tCol < N) {
                C[(cRow + tRow + 9) * N + cCol + tCol] = acc[m][n][3];
            }
        }
    }
#endif
}

// ============================================================================
// Class Implementation
// ============================================================================

MatmulMmaBf16::MatmulMmaBf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        throw std::runtime_error("MMA BF16 requires SM 8.0+");
    }

    if (N % 128 != 0) {
        throw std::runtime_error("N must be multiple of 128");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulMmaBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    // Convert to BF16
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16Mma<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16Mma<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    // Calculate shared memory size
    size_t smemSize = NUM_STAGES * (BM * BK + BK * BN) * sizeof(__nv_bfloat16);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    cudaFuncSetAttribute(matmulMmaBf16Kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smemSize);

    matmulMmaBf16Kernel<<<blocks, threads, smemSize>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulMmaBf16::~MatmulMmaBf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
