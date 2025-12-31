#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
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

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N    Set array size (default: 1048576)\n");
    printf("  -v, --verify    Enable result verification\n");
    printf("  -h, --help      Show this help message\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments using getopt_long
    bool verify = false;
    int n = 1 << 20;  // Default: 1 million elements

    static struct option long_options[] = {
        {"size",   required_argument, 0, 'n'},
        {"verify", no_argument,       0, 'v'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:vh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                if (n <= 0) {
                    fprintf(stderr, "Error: size must be positive\n");
                    return 1;
                }
                break;
            case 'v':
                verify = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
        }
    }

    // Problem size
    size_t bytes = n * sizeof(float);

    printf("Vector addition of %d elements\n", n);
    if (verify) {
        printf("Verification enabled\n");
    }

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize random seed
    srand(time(NULL));

    // Initialize input vectors with random values
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
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

    // Create CUDA events for timing kernel execution
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    // Run kernel multiple times to collect statistics
    const int num_iterations = 1000;
    float *timings = (float*)malloc(num_iterations * sizeof(float));

    printf("Running kernel %d times to collect statistics...\n", num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(kernel_start);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
        cudaEventRecord(kernel_stop);

        cudaEventSynchronize(kernel_stop);
        cudaEventElapsedTime(&timings[i], kernel_start, kernel_stop);
    }

    // Check for kernel launch errors
    cudaCheckError(cudaGetLastError());

    // Calculate statistics
    float min_time = timings[0];
    float max_time = timings[0];
    float sum_time = 0.0f;

    for (int i = 0; i < num_iterations; i++) {
        if (timings[i] < min_time) min_time = timings[i];
        if (timings[i] > max_time) max_time = timings[i];
        sum_time += timings[i];
    }
    float avg_time = sum_time / num_iterations;

    // Sort for percentiles
    qsort(timings, num_iterations, sizeof(float), [](const void *a, const void *b) {
        float fa = *(const float*)a;
        float fb = *(const float*)b;
        return (fa > fb) - (fa < fb);
    });

    // Calculate percentiles
    int p50_idx = (int)(num_iterations * 0.50);
    int p90_idx = (int)(num_iterations * 0.90);
    int p95_idx = (int)(num_iterations * 0.95);
    int p99_idx = (int)(num_iterations * 0.99);

    printf("\n===========================================\n");
    printf("Kernel Execution Statistics (%d runs):\n", num_iterations);
    printf("===========================================\n");
    printf("  Min:    %.3f ms\n", min_time);
    printf("  Max:    %.3f ms\n", max_time);
    printf("  Mean:   %.3f ms\n", avg_time);
    printf("  Median: %.3f ms\n", timings[p50_idx]);
    printf("  P90:    %.3f ms\n", timings[p90_idx]);
    printf("  P95:    %.3f ms\n", timings[p95_idx]);
    printf("  P99:    %.3f ms\n", timings[p99_idx]);
    printf("===========================================\n");

    free(timings);

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    printf("Vector addition completed successfully!\n");

    // Verify result if requested
    if (verify) {
        printf("Verifying results...\n");
        bool success = true;
        for (int i = 0; i < n; i++) {
            float expected = h_a[i] + h_b[i];
            if (fabs(h_c[i] - expected) > 1e-5) {
                fprintf(stderr, "Verification failed at element %d: expected %f, got %f\n",
                        i, expected, h_c[i]);
                success = false;
                break;
            }
        }
        if (success) {
            printf("Verification PASSED! All %d elements are correct.\n", n);
        }
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Destroy events
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);

    printf("\nCongratulations! Your first CUDA program ran successfully!\n");

    return 0;
}
