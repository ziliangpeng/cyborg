#include "matmul_wgmma_bf16.h"
#include "cuda_utils.h"
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>

// ============================================================================
// WGMMA Configuration
// Using m64n64k16 instruction - each warpgroup computes 64x64 output
// This uses 32 output registers per thread, which is manageable
//
// NOTE: This is an experimental kernel using direct PTX WGMMA instructions.
// WGMMA requires very specific memory layouts (swizzled tiles, not row-major).
// Without proper swizzled layouts, correctness is not guaranteed.
// For production use, consider using CUTLASS which handles these layouts.
// ============================================================================

#define BM 64           // Block tile M (matches WGMMA M=64)
#define BN 64           // Block tile N (matches WGMMA N=64)
#define BK 16           // Block tile K (matches WGMMA K=16)

// Warpgroup = 4 warps = 128 threads
#define WARPGROUP_SIZE 128
#define BLOCK_SIZE WARPGROUP_SIZE  // One warpgroup per block for simplicity

// Shared memory with swizzle-friendly padding
// For wgmma, we need specific layouts
#define SMEM_A_STRIDE (BK + 8)     // Padding for bank conflict avoidance
#define SMEM_B_STRIDE (BN + 8)     // 64 + 8 = 72

// ============================================================================
// Matrix Descriptor Creation for WGMMA
// Based on NVIDIA PTX ISA and Modular documentation
// Bit layout:
//   Bits 0-13:  Base address (14 bits, values are >> 4, ignoring 4 LSB)
//   Bits 16-29: Leading dimension byte offset (LBO, 14 bits, >> 4)
//   Bits 32-45: Stride dimension byte offset (SBO, 14 bits, >> 4)
//   Bits 62-63: Swizzle mode (0=128B, 1=64B, 2=32B, 3=NONE/interleave)
// ============================================================================

// Swizzle mode encodings
#define SWIZZLE_128B 0
#define SWIZZLE_64B  1
#define SWIZZLE_32B  2
#define SWIZZLE_NONE 3

__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr, int leading_dim_bytes, int stride_bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;

    // Start address (bits 0-13): address >> 4 (16-byte aligned)
    desc |= ((uint64_t)(addr >> 4) & 0x3FFF);

    // Leading dimension byte offset (bits 16-29): LBO >> 4
    desc |= ((uint64_t)(leading_dim_bytes >> 4) & 0x3FFF) << 16;

    // Stride dimension byte offset (bits 32-45): SBO >> 4
    desc |= ((uint64_t)(stride_bytes >> 4) & 0x3FFF) << 32;

    // Swizzle mode (bits 62-63): use no swizzle for simple row-major layout
    desc |= ((uint64_t)SWIZZLE_NONE) << 62;

    return desc;
}

// ============================================================================
// WGMMA PTX Wrappers
// ============================================================================

// Fence before WGMMA operations
__device__ __forceinline__ void wgmma_fence() {
#if __CUDA_ARCH__ >= 900
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

// Commit a group of WGMMA operations
__device__ __forceinline__ void wgmma_commit_group() {
#if __CUDA_ARCH__ >= 900
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}

// Wait for WGMMA operations to complete
__device__ __forceinline__ void wgmma_wait_group() {
#if __CUDA_ARCH__ >= 900
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}

// WGMMA m64n64k16 BF16 with FP32 accumulator
// A: 64x16 BF16 in shared memory
// B: 16x64 BF16 in shared memory
// C: 64x64 FP32 in registers (distributed across warpgroup)
// Each thread holds 64*64/128 = 32 floats = 32 registers for f32 accumulator
__device__ __forceinline__ void wgmma_m64n64k16_bf16_f32(
    uint64_t desc_a, uint64_t desc_b,
    float& d00, float& d01, float& d02, float& d03,
    float& d04, float& d05, float& d06, float& d07,
    float& d08, float& d09, float& d10, float& d11,
    float& d12, float& d13, float& d14, float& d15,
    float& d16, float& d17, float& d18, float& d19,
    float& d20, float& d21, float& d22, float& d23,
    float& d24, float& d25, float& d26, float& d27,
    float& d28, float& d29, float& d30, float& d31,
    int scale_d = 1)
{
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " p, 1, 1, 0, 0;\n"  // scale_d, scale_a=1, scale_b=1, tnsp_a=0, tnsp_b=0
        "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03),
          "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
          "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
          "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
          "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19),
          "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
          "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27),
          "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        : "l"(desc_a), "l"(desc_b), "r"(scale_d));
#endif
}

// ============================================================================
// FP32 to BF16 Conversion
// ============================================================================

__global__ void convertFP32ToBF16Wgmma(const float* input, __nv_bfloat16* output, int size) {
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
// WGMMA BF16 Kernel
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
matmulWgmmaBf16Kernel(const __nv_bfloat16* __restrict__ A,
                      const __nv_bfloat16* __restrict__ B,
                      float* __restrict__ C,
                      int N) {
#if __CUDA_ARCH__ >= 900
    // Shared memory for A and B tiles
    // A: BM x BK = 64 x 16, stored in row-major for K-major WGMMA
    // B: BK x BN = 16 x 128, stored in row-major
    extern __shared__ char smem[];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Bs = As + BM * SMEM_A_STRIDE;

    const int blockM = blockIdx.y * BM;
    const int blockN = blockIdx.x * BN;

    const int tid = threadIdx.x;

    // Initialize accumulators to zero
    // Each thread in warpgroup holds part of the 64x64 output
    // m64n64k16 distributes 64*64 = 4096 floats across 128 threads = 32 floats per thread
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }

    const int numKTiles = N / BK;

    // Main loop over K dimension
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int kOffset = kTile * BK;

        // Cooperative load of A tile (64 x 16)
        // 128 threads load 64*16 = 1024 elements = 8 elements per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int linearIdx = tid + i * BLOCK_SIZE;
            int row = linearIdx / BK;
            int col = linearIdx % BK;
            if (row < BM && (kOffset + col) < N) {
                As[row * SMEM_A_STRIDE + col] = A[(blockM + row) * N + kOffset + col];
            }
        }

        // Cooperative load of B tile (16 x 64)
        // 128 threads load 16*64 = 1024 elements = 8 elements per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int linearIdx = tid + i * BLOCK_SIZE;
            int row = linearIdx / BN;
            int col = linearIdx % BN;
            if (row < BK && (blockN + col) < N) {
                Bs[row * SMEM_B_STRIDE + col] = B[(kOffset + row) * N + blockN + col];
            }
        }

        __syncthreads();

        // Create matrix descriptors for WGMMA
        uint64_t desc_a = make_smem_desc(As, SMEM_A_STRIDE * sizeof(__nv_bfloat16), BK * sizeof(__nv_bfloat16));
        uint64_t desc_b = make_smem_desc(Bs, SMEM_B_STRIDE * sizeof(__nv_bfloat16), BN * sizeof(__nv_bfloat16));

        // Issue WGMMA
        wgmma_fence();

        wgmma_m64n64k16_bf16_f32(
            desc_a, desc_b,
            acc[0],  acc[1],  acc[2],  acc[3],
            acc[4],  acc[5],  acc[6],  acc[7],
            acc[8],  acc[9],  acc[10], acc[11],
            acc[12], acc[13], acc[14], acc[15],
            acc[16], acc[17], acc[18], acc[19],
            acc[20], acc[21], acc[22], acc[23],
            acc[24], acc[25], acc[26], acc[27],
            acc[28], acc[29], acc[30], acc[31],
            1);

        wgmma_commit_group();
        wgmma_wait_group();

        __syncthreads();
    }

    // Store results
    // The accumulator layout for m64n64 across 128 threads:
    // Each thread holds 32 floats representing its portion of the 64x64 tile
    // Total: 128 threads * 32 floats = 4096 floats = 64*64

    // WGMMA m64n64 output mapping:
    // The 128 threads are organized as 4 warps of 32 threads
    // Each warp handles a 16x64 portion of the 64x64 output
    // Within each warp, the output follows the NVIDIA accumulator layout

    // Simplified mapping: distribute threads across the output tile
    // Thread i owns elements at specific positions based on wgmma layout
    const int warpId = tid / 32;
    const int laneId = tid % 32;

    // Based on WGMMA output layout documentation:
    // For m64n64, each thread owns 32 consecutive elements in a specific pattern
    // Warp mapping: warp 0 -> rows 0-15, warp 1 -> rows 16-31, etc.
    // Lane mapping within warp follows replicated Z-pattern

    // Simple linear mapping (may not match exact WGMMA layout)
    // Each thread writes 32 values: 4 rows * 8 columns
    int baseRow = warpId * 16 + (laneId / 8) * 4;
    int baseCol = (laneId % 8) * 8;

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int outRow = blockM + baseRow + r;
        if (outRow < N) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int outCol = blockN + baseCol + c;
                if (outCol < N) {
                    C[outRow * N + outCol] = acc[r * 8 + c];
                }
            }
        }
    }
#endif
}

// ============================================================================
// Class Implementation
// ============================================================================

MatmulWgmmaBf16::MatmulWgmmaBf16(int N, int blockDim) : N(N) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 9) {
        throw std::runtime_error("WGMMA BF16 requires SM 9.0+ (Hopper)");
    }

    if (N % 64 != 0) {
        throw std::runtime_error("N must be multiple of 64 for WGMMA kernel");
    }

    cudaCheckError(cudaMalloc(&d_A_bf16, N * N * sizeof(__nv_bfloat16)));
    cudaCheckError(cudaMalloc(&d_B_bf16, N * N * sizeof(__nv_bfloat16)));
}

void MatmulWgmmaBf16::execute(const float *d_A, const float *d_B, float *d_C) {
    // Convert FP32 to BF16
    int convThreads = 256;
    int convBlocks = (N * N / 4 + convThreads - 1) / convThreads;
    convertFP32ToBF16Wgmma<<<convBlocks, convThreads>>>(d_A, d_A_bf16, N * N);
    convertFP32ToBF16Wgmma<<<convBlocks, convThreads>>>(d_B, d_B_bf16, N * N);

    // Launch WGMMA kernel
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    // Calculate shared memory size
    size_t smem_size = (BM * SMEM_A_STRIDE + BK * SMEM_B_STRIDE) * sizeof(__nv_bfloat16);

    matmulWgmmaBf16Kernel<<<blocks, threads, smem_size>>>(d_A_bf16, d_B_bf16, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulWgmmaBf16::~MatmulWgmmaBf16() {
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
}
