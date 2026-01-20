// pybind11 wrapper for softmax CUDA kernel using C API
//
// This module exposes the OnlineWarpSoftmax CUDA kernel to Python.
// The interface uses NumPy arrays for input/output.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "softmax_c_api.h"

namespace py = pybind11;

// Softmax using OnlineWarpSoftmax kernel
// Input: 1D float32 NumPy array
// Output: 1D float32 NumPy array (same shape)
py::array_t<float> softmax_cuda(py::array_t<float> input, int block_size = 256) {
    // Request buffer info for input
    py::buffer_info buf_in = input.request();

    // Validate input
    if (buf_in.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }

    int n = buf_in.shape[0];
    float* host_input = static_cast<float*>(buf_in.ptr);

    // Allocate device memory
    float* d_input = softmax_alloc_device(n);
    float* d_output = softmax_alloc_device(n);

    // Copy input to device
    softmax_copy_to_device(d_input, host_input, n);

    // Create kernel instance and execute
    SoftmaxHandle kernel = softmax_create(n, block_size);
    softmax_execute(kernel, d_input, d_output);
    softmax_destroy(kernel);

    // Synchronize to ensure kernel completion
    softmax_sync();

    // Allocate output array and copy result back
    auto result = py::array_t<float>(n);
    py::buffer_info buf_out = result.request();
    float* host_output = static_cast<float*>(buf_out.ptr);

    softmax_copy_to_host(host_output, d_output, n);

    // Free device memory
    softmax_free_device(d_input);
    softmax_free_device(d_output);

    return result;
}

// Timed softmax for benchmarking (returns output and time in ms)
std::tuple<py::array_t<float>, float> softmax_cuda_timed(py::array_t<float> input, int block_size = 256) {
    py::buffer_info buf_in = input.request();

    if (buf_in.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }

    int n = buf_in.shape[0];
    float* host_input = static_cast<float*>(buf_in.ptr);

    // Allocate device memory
    float* d_input = softmax_alloc_device(n);
    float* d_output = softmax_alloc_device(n);

    // Copy input to device
    softmax_copy_to_device(d_input, host_input, n);

    // Create kernel instance
    SoftmaxHandle kernel = softmax_create(n, block_size);

    // Create CUDA events for timing
    CudaEvent start = softmax_event_create();
    CudaEvent stop = softmax_event_create();

    // Time the kernel execution only (not memory transfers)
    softmax_event_record(start);
    softmax_execute(kernel, d_input, d_output);
    softmax_event_record(stop);
    softmax_event_sync(stop);

    float time_ms = softmax_event_elapsed(start, stop);

    // Cleanup kernel and events
    softmax_destroy(kernel);
    softmax_event_destroy(start);
    softmax_event_destroy(stop);

    // Allocate output array and copy result back
    auto result = py::array_t<float>(n);
    py::buffer_info buf_out = result.request();
    float* host_output = static_cast<float*>(buf_out.ptr);

    softmax_copy_to_host(host_output, d_output, n);

    // Free device memory
    softmax_free_device(d_input);
    softmax_free_device(d_output);

    return std::make_tuple(result, time_ms);
}

// Benchmark function: run kernel multiple times and return list of times
std::vector<float> softmax_cuda_benchmark(py::array_t<float> input, int iterations = 100, int warmup = 10, int block_size = 256) {
    py::buffer_info buf_in = input.request();

    if (buf_in.ndim != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }

    int n = buf_in.shape[0];
    float* host_input = static_cast<float*>(buf_in.ptr);

    // Allocate device memory
    float* d_input = softmax_alloc_device(n);
    float* d_output = softmax_alloc_device(n);

    // Copy input to device
    softmax_copy_to_device(d_input, host_input, n);

    // Create kernel instance
    SoftmaxHandle kernel = softmax_create(n, block_size);

    // Create CUDA events for timing
    CudaEvent start = softmax_event_create();
    CudaEvent stop = softmax_event_create();

    // Warmup iterations
    for (int i = 0; i < warmup; i++) {
        softmax_execute(kernel, d_input, d_output);
    }
    softmax_sync();

    // Benchmark iterations with per-iteration timing
    std::vector<float> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        softmax_event_record(start);
        softmax_execute(kernel, d_input, d_output);
        softmax_event_record(stop);
        softmax_event_sync(stop);

        float time_ms = softmax_event_elapsed(start, stop);
        times.push_back(time_ms);
    }

    // Cleanup
    softmax_destroy(kernel);
    softmax_event_destroy(start);
    softmax_event_destroy(stop);
    softmax_free_device(d_input);
    softmax_free_device(d_output);

    return times;
}

PYBIND11_MODULE(softmax_cuda, m) {
    m.doc() = "CUDA Softmax (OnlineWarp) - pybind11 bindings";

    m.def("softmax", &softmax_cuda,
          "Compute softmax using CUDA OnlineWarp kernel",
          py::arg("input"),
          py::arg("block_size") = 256);

    m.def("softmax_timed", &softmax_cuda_timed,
          "Compute softmax and return (output, time_ms) tuple",
          py::arg("input"),
          py::arg("block_size") = 256);

    m.def("benchmark", &softmax_cuda_benchmark,
          "Run softmax kernel multiple times and return list of timing results (ms)",
          py::arg("input"),
          py::arg("iterations") = 100,
          py::arg("warmup") = 10,
          py::arg("block_size") = 256);
}
