#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "vector_kernels.h"
#include "vector_init.h"

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N         Set array size (default: 1048576)\n");
    printf("  -b, --block-size N   Set threads per block (default: 256)\n");
    printf("  -v, --verify         Enable result verification\n");
    printf("  -h, --help           Show this help message\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    bool verify = false;
    int n = 1 << 20;  // Default: 1 million elements
    // Default: 256 threads per block (NVIDIA recommended, 8 warps, optimal for most kernels)
    int threadsPerBlock = 256;

    static struct option long_options[] = {
        {"size",       required_argument, 0, 'n'},
        {"block-size", required_argument, 0, 'b'},
        {"verify",     no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:b:vh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                if (n <= 0) {
                    fprintf(stderr, "Error: size must be positive\n");
                    return 1;
                }
                break;
            case 'b':
                threadsPerBlock = atoi(optarg);
                if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
                    fprintf(stderr, "Error: block-size must be between 1 and 1024\n");
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

    printf("Vector addition of %d elements\n", n);
    if (verify) {
        printf("Verification enabled\n");
    }

    // Initialize random seed
    srand(time(NULL));

    // Allocate and initialize host vectors
    float *h_a, *h_b, *h_c;
    allocateAndInitVector(&h_a, n);
    allocateAndInitVector(&h_b, n);
    allocateAndInitVector(&h_c, n);

    // Allocate device vectors
    float *d_a, *d_b, *d_c;
    allocateDeviceVector(&d_a, n);
    allocateDeviceVector(&d_b, n);
    allocateDeviceVector(&d_c, n);

    // Transfer input vectors to device
    transferToDevice(d_a, h_a, n);
    transferToDevice(d_b, h_b, n);

    // Configure kernel launch parameters
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks and %d threads per block\n",
           blocksPerGrid, threadsPerBlock);

    // Create CUDA events for timing
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

    // Transfer result back to host
    transferFromDevice(h_c, d_c, n);
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

    // Cleanup
    freeDeviceVector(d_a);
    freeDeviceVector(d_b);
    freeDeviceVector(d_c);
    freeHostVector(h_a);
    freeHostVector(h_b);
    freeHostVector(h_c);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);

    printf("\nCongratulations! Your first CUDA program ran successfully!\n");

    return 0;
}
