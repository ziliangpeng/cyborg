#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>

// H100 theoretical memory bandwidth (from spec sheet)
#define H100_MEMORY_BANDWIDTH_GBS 3350.0f  // 3.35 TB/s

// Wrapper for cudaMemcpy that measures and reports bandwidth
// label is optional - pass NULL to use direction only
inline cudaError_t cudaMemcpyTimed(void *dst, const void *src, size_t count,
                                    cudaMemcpyKind kind, const char *label) {
    // Generate direction string based on copy kind
    const char *direction;
    switch (kind) {
        case cudaMemcpyHostToDevice:
            direction = "Host -> Device";
            break;
        case cudaMemcpyDeviceToHost:
            direction = "Device -> Host";
            break;
        case cudaMemcpyDeviceToDevice:
            direction = "Device -> Device";
            break;
        case cudaMemcpyHostToHost:
            direction = "Host -> Host";
            break;
        default:
            direction = "Memory Copy";
            break;
    }

    // Build full label
    char full_label[256];
    if (label) {
        snprintf(full_label, sizeof(full_label), "%s (%s)", direction, label);
    } else {
        snprintf(full_label, sizeof(full_label), "%s", direction);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    // Calculate bandwidth in GB/s
    float bandwidth_gbs = (count / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
    float efficiency = (bandwidth_gbs / H100_MEMORY_BANDWIDTH_GBS) * 100.0f;

    printf("  %s: %.2f MB in %.3f ms -> %.2f GB/s (%.2f%% of peak)\n",
           full_label,
           count / (1024.0f * 1024.0f),
           time_ms,
           bandwidth_gbs,
           efficiency);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return err;
}

#endif // CUDA_UTILS_H
