#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel: runs on the GPU
// Each thread computes one element of the result
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate global thread ID
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Helper function to check CUDA errors
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

int main() {
    // Problem size
    int n = 1 << 20;  // 1 million elements
    size_t bytes = n * sizeof(float);

    printf("Vector addition of %d elements\n", n);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaCheckError(cudaMalloc(&d_a, bytes));
    cudaCheckError(cudaMalloc(&d_b, bytes));
    cudaCheckError(cudaMalloc(&d_c, bytes));

    // Copy data from host to device
    cudaCheckError(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    // Use 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel with %d blocks and %d threads per block\n",
           blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Check for kernel launch errors
    cudaCheckError(cudaGetLastError());

    // Wait for GPU to finish
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify result
    printf("Verifying result...\n");
    for (int i = 0; i < n; i++) {
        if (fabs(h_c[i] - 3.0f) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED! All %d elements correctly computed.\n", n);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\nCongratulations! Your first CUDA program ran successfully!\n");

    return 0;
}
