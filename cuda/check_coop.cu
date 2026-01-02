#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Check cooperative launch support
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
    printf("Cooperative Launch Support: %s\n", supportsCoopLaunch ? "YES" : "NO");

    // Check max blocks per multiprocessor for grid sizing
    int maxBlocksPerSM = 0;
    cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, device);
    printf("Max Blocks per SM: %d\n", maxBlocksPerSM);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max cooperative grid size: ~%d blocks\n", maxBlocksPerSM * prop.multiProcessorCount);

    return 0;
}
