# TODO

## CUPTI Integration
- [ ] Try using CUPTI (CUDA Profiling Tools Interface) directly
  - Explore Callback API for hooking into CUDA API calls
  - Test Activity API for collecting GPU events asynchronously
  - Experiment with Metrics API for hardware counters
  - Check CUPTI samples in `$CUDA_HOME/extras/CUPTI/samples/`
  - References:
    - [CUPTI Documentation](https://docs.nvidia.com/cupti/)
    - [CUPTI Tutorial GitHub](https://github.com/eunomia-bpf/cupti-tutorial)
    - [How to build a CUDA profiler](https://conless.dev/blog/2025/cupti-profiler/)

## Future Improvements
- [ ] Add more command-line flags (e.g., threads per block, number of iterations)
- [ ] Implement different CUDA kernels (matrix multiplication, reduction, etc.)
- [ ] Compare performance with CPU implementation
- [ ] Experiment with different memory patterns (coalesced vs uncoalesced)
- [ ] Try unified memory vs explicit transfers
