#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "batch_softmax_naive.h"
#include "batch_softmax_warp.h"
#include "vector_init.h"
#include "common/benchmark/benchmark_utils.h"

// ============================================================================
// BENCHMARK MODE: Data Structures and Constants
// ============================================================================

// Benchmark configuration
const char* BENCHMARK_METHODS[] = {
    "naive",
    "warp"
};
const int NUM_METHODS = 2;

// Benchmark sizes: (batch_size, dim) pairs
struct BenchmarkSize {
    int batch;
    int dim;
    const char* label;
};

const BenchmarkSize BENCHMARK_SIZES[] = {
    {64, 64, "64x64"},
    {64, 256, "64x256"},
    {64, 1024, "64x1K"},
    {256, 256, "256x256"},
    {256, 1024, "256x1K"},
    {1024, 256, "1Kx256"},
    {1024, 1024, "1Kx1K"},
    {4096, 256, "4Kx256"},
    {4096, 1024, "4Kx1K"},
    {8192, 512, "8Kx512"},
};
const int NUM_SIZES = sizeof(BENCHMARK_SIZES) / sizeof(BENCHMARK_SIZES[0]);

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -b, --batch N             Set batch size (default: 64)\n");
    printf("  -d, --dim N               Set dimension per row (default: 1024)\n");
    printf("  -t, --threads N           Set threads per block (default: 256)\n");
    printf("  -m, --method METHOD       Method: 'naive', 'warp', or 'all' (default: warp)\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nMethods:\n");
    printf("  naive:  One block per row, shared memory reductions (works for any dim)\n");
    printf("  warp:   One warp per row, warp shuffle reductions (optimal for small dims)\n");
    printf("\nSpecial method:\n");
    printf("  all:    Run comprehensive benchmark across all methods and sizes\n");
    printf("          Tests various (batch, dim) combinations\n");
    printf("          Iterations: 100 per test\n");
    printf("          Output: Formatted performance table\n");
    printf("\nExample usage:\n");
    printf("  %s --method all                     # Performance benchmark\n", program_name);
    printf("  %s --batch 256 --dim 512 --method naive  # Single test\n", program_name);
}

// ============================================================================
// CPU Reference Implementation for Verification
// ============================================================================

void cpu_batch_softmax(const float *input, float *output, int batch_size, int dim) {
    for (int row = 0; row < batch_size; row++) {
        const float *row_in = input + row * dim;
        float *row_out = output + row * dim;

        // Find max
        float max_val = row_in[0];
        for (int i = 1; i < dim; i++) {
            max_val = fmaxf(max_val, row_in[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            row_out[i] = expf(row_in[i] - max_val);
            sum += row_out[i];
        }

        // Normalize
        for (int i = 0; i < dim; i++) {
            row_out[i] /= sum;
        }
    }
}

// ============================================================================
// Verification Helper
// ============================================================================

bool verify_output(const float *gpu_output, const float *cpu_output, int batch_size, int dim, float tolerance = 1e-5f) {
    int total = batch_size * dim;
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    for (int i = 0; i < total; i++) {
        float diff = fabsf(gpu_output[i] - cpu_output[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    if (max_diff > tolerance) {
        int row = max_diff_idx / dim;
        int col = max_diff_idx % dim;
        printf("  Verification FAILED: max diff %.6f at [%d, %d] (GPU: %.6f, CPU: %.6f)\n",
               max_diff, row, col, gpu_output[max_diff_idx], cpu_output[max_diff_idx]);
        return false;
    }
    return true;
}

// ============================================================================
// BENCHMARK MODE: Main Benchmark Function
// ============================================================================

void benchmark_all_methods(int threadsPerBlock) {
    printf("\n=============================================================================\n");
    printf("                    RUNNING COMPREHENSIVE BENCHMARK\n");
    printf("=============================================================================\n");
    printf("Methods to test: %d\n", NUM_METHODS);
    printf("Sizes to test: %d\n", NUM_SIZES);
    printf("Iterations per test: 100\n");
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("=============================================================================\n\n");

    // Convert to vectors for the utility functions
    std::vector<std::string> method_names(BENCHMARK_METHODS, BENCHMARK_METHODS + NUM_METHODS);
    std::vector<std::string> size_labels;
    for (int i = 0; i < NUM_SIZES; i++) {
        size_labels.push_back(BENCHMARK_SIZES[i].label);
    }

    // Allocate result arrays
    std::vector<std::vector<BenchmarkResult>> all_results(NUM_METHODS, std::vector<BenchmarkResult>(NUM_SIZES));

    BenchmarkConfig config;
    config.warmup_iterations = 10;
    config.timed_iterations = 100;

    // Run benchmarks for each method and size
    for (int m = 0; m < NUM_METHODS; m++) {
        const char *method = BENCHMARK_METHODS[m];
        printf("Testing method: %s\n", method);

        for (int s = 0; s < NUM_SIZES; s++) {
            int batch_size = BENCHMARK_SIZES[s].batch;
            int dim = BENCHMARK_SIZES[s].dim;
            int total = batch_size * dim;
            printf("  Size: %s (%d x %d = %d elements)... ", BENCHMARK_SIZES[s].label, batch_size, dim, total);
            fflush(stdout);

            // Allocate host memory
            float *h_input;
            allocateAndInitVector(&h_input, total);

            // Allocate device memory
            float *d_input, *d_output;
            cudaError_t err = cudaMalloc(&d_input, total * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                all_results[m][s] = BenchmarkResult("Device allocation failed");
                freeHostVector(h_input);
                continue;
            }

            err = cudaMalloc(&d_output, total * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                all_results[m][s] = BenchmarkResult("Device allocation failed");
                freeHostVector(h_input);
                cudaFree(d_input);
                continue;
            }

            cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);

            // Try to instantiate kernel
            BatchSoftmaxKernel *kernel = nullptr;
            try {
                if (strcmp(method, "naive") == 0) {
                    kernel = new NaiveBatchSoftmax(batch_size, dim, threadsPerBlock);
                } else if (strcmp(method, "warp") == 0) {
                    kernel = new WarpBatchSoftmax(batch_size, dim, threadsPerBlock);
                }

                if (!kernel) {
                    printf("SKIPPED (unknown method)\n");
                    all_results[m][s] = BenchmarkResult("Unknown method");
                } else {
                    // Run performance benchmark with per-iteration timing
                    auto kernel_fn = [&]() { kernel->execute(d_input, d_output); };
                    TimingStats stats = get_timing_stats(kernel_fn, config);

                    all_results[m][s] = BenchmarkResult(stats.p50_ms, stats.p90_ms);
                    printf("%.3f/%.3f ms\n", stats.p50_ms, stats.p90_ms);

                    delete kernel;
                }
            } catch (const std::exception &e) {
                printf("SKIPPED (exception: %s)\n", e.what());
                all_results[m][s] = BenchmarkResult(std::string("Exception: ") + e.what());
                if (kernel) delete kernel;
            } catch (...) {
                printf("SKIPPED (unknown exception)\n");
                all_results[m][s] = BenchmarkResult("Unknown exception");
                if (kernel) delete kernel;
            }

            // Cleanup
            freeHostVector(h_input);
            cudaFree(d_input);
            cudaFree(d_output);
        }
        printf("\n");
    }

    // Print results using utility functions
    print_benchmark_header("BATCH SOFTMAX PERFORMANCE BENCHMARK", size_labels, config.timed_iterations);
    for (int m = 0; m < NUM_METHODS; m++) {
        print_benchmark_row(BENCHMARK_METHODS[m], all_results[m]);
    }
    print_benchmark_footer(method_names, all_results);
    print_top_fastest(method_names, all_results, size_labels, 2, {});
}

// ============================================================================
// Single Method Benchmark
// ============================================================================

void batch_softmax_op(int batch_size, int dim, int threadsPerBlock, const char *method) {
    int total = batch_size * dim;
    printf("Batch Softmax: %d rows x %d columns = %d elements\n", batch_size, dim, total);
    printf("Method: %s\n", method);
    printf("Threads per block: %d\n", threadsPerBlock);

    // Allocate and initialize input
    float *h_input;
    allocateAndInitVector(&h_input, total);

    // Allocate host output for verification
    float *h_output = (float*)malloc(total * sizeof(float));
    float *h_output_cpu = (float*)malloc(total * sizeof(float));
    if (!h_output || !h_output_cpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        freeHostVector(h_input);
        return;
    }

    // Compute CPU reference
    cpu_batch_softmax(h_input, h_output_cpu, batch_size, dim);

    // Allocate device vectors
    float *d_input, *d_output;
    allocateDeviceVector(&d_input, total);
    allocateDeviceVector(&d_output, total);
    transferToDevice(d_input, h_input, total);

    // Configure timing
    const int num_iterations = 100;
    float *timings = (float*)malloc(num_iterations * sizeof(float));
    if (!timings) {
        fprintf(stderr, "Failed to allocate memory for timings\n");
        freeHostVector(h_input);
        free(h_output);
        free(h_output_cpu);
        freeDeviceVector(d_input);
        freeDeviceVector(d_output);
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running batch softmax %d times to collect statistics...\n", num_iterations);

    // Instantiate kernel
    BatchSoftmaxKernel *kernel = nullptr;
    if (strcmp(method, "naive") == 0) {
        kernel = new NaiveBatchSoftmax(batch_size, dim, threadsPerBlock);
    } else if (strcmp(method, "warp") == 0) {
        kernel = new WarpBatchSoftmax(batch_size, dim, threadsPerBlock);
    }

    if (kernel) {
        // Time kernel execution
        for (int i = 0; i < num_iterations; i++) {
            cudaEventRecord(start);
            kernel->execute(d_input, d_output);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&timings[i], start, stop);
        }
        delete kernel;
    } else {
        fprintf(stderr, "Unknown method: %s\n", method);
        freeHostVector(h_input);
        free(h_output);
        free(h_output_cpu);
        freeDeviceVector(d_input);
        freeDeviceVector(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(timings);
        return;
    }

    cudaCheckError(cudaGetLastError());

    // Calculate and print statistics
    calculate_and_print_statistics(timings, num_iterations);

    // Verify correctness
    transferFromDevice(h_output, d_output, total);
    printf("\nVerifying correctness against CPU reference...\n");
    if (verify_output(h_output, h_output_cpu, batch_size, dim)) {
        printf("  Verification PASSED\n");
    }

    // Cleanup
    freeDeviceVector(d_input);
    freeDeviceVector(d_output);
    freeHostVector(h_input);
    free(h_output);
    free(h_output_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nBatch Softmax completed successfully!\n");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    const char *method = "warp";  // Default method
    int batch_size = 64;
    int dim = 1024;
    int threadsPerBlock = 256;

    static struct option long_options[] = {
        {"batch",   required_argument, 0, 'b'},
        {"dim",     required_argument, 0, 'd'},
        {"threads", required_argument, 0, 't'},
        {"method",  required_argument, 0, 'm'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "b:d:t:m:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'b':
                batch_size = atoi(optarg);
                if (batch_size <= 0) {
                    fprintf(stderr, "Error: batch size must be positive\n");
                    return 1;
                }
                break;
            case 'd':
                dim = atoi(optarg);
                if (dim <= 0) {
                    fprintf(stderr, "Error: dimension must be positive\n");
                    return 1;
                }
                break;
            case 't':
                threadsPerBlock = atoi(optarg);
                if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
                    fprintf(stderr, "Error: threads must be between 1 and 1024\n");
                    return 1;
                }
                break;
            case 'm':
                method = optarg;
                if (strcmp(method, "naive") != 0 && strcmp(method, "warp") != 0 &&
                    strcmp(method, "all") != 0) {
                    fprintf(stderr, "Error: method must be 'naive', 'warp', or 'all'\n");
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
        }
    }

    // Initialize random seed
    srand(time(NULL));

    // Run batch softmax operation
    if (strcmp(method, "all") == 0) {
        // Benchmark mode: test all methods across all sizes
        benchmark_all_methods(threadsPerBlock);
    } else {
        // Single method mode
        batch_softmax_op(batch_size, dim, threadsPerBlock, method);
    }

    return 0;
}
