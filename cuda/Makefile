# Makefile for CUDA programs

# CUDA compiler
NVCC = nvcc

# Compiler flags (can be overridden: make NVCC_FLAGS="-O2 -arch=sm_86")
NVCC_FLAGS ?= -O2 -arch=sm_90

# Targets
TARGETS = vector mem_benchmark

# Default target - build all
all: $(TARGETS)

# Build vector from multiple source files
vector: vector.cu vector_kernels.cu vector_init.cu cuda_utils.h vector_kernels.h vector_init.h
	$(NVCC) $(NVCC_FLAGS) vector.cu vector_kernels.cu vector_init.cu -o vector

# Build mem_benchmark
mem_benchmark: mem_benchmark.cu cuda_utils.h
	$(NVCC) $(NVCC_FLAGS) mem_benchmark.cu -o mem_benchmark

# Run vector
run: vector
	./vector

# Test all binaries
test: all
	@echo "=========================================="
	@echo "Running verification tests..."
	@echo "=========================================="
	@echo "Testing vector..."
	@./vector -n 1000 -v > /dev/null && echo "  ✓ 1K elements"
	@./vector -n 1000000 -v > /dev/null && echo "  ✓ 1M elements"
	@./vector -n 10000000 -v > /dev/null && echo "  ✓ 10M elements"
	@echo ""
	@echo "Testing mem_benchmark..."
	@./mem_benchmark -n 10 -i 1 > /dev/null && echo "  ✓ Pageable (10MB)"
	@./mem_benchmark -n 10 -p -i 1 > /dev/null && echo "  ✓ Pinned (10MB)"
	@echo ""
	@echo "=========================================="
	@echo "All tests passed!"
	@echo "=========================================="

# Clean build artifacts
clean:
	rm -f $(TARGETS) *.o

# Check if CUDA is available
check:
	@echo "Checking CUDA installation..."
	@which nvcc >/dev/null || (echo "ERROR: nvcc not found. Please install CUDA toolkit." && exit 1)
	@nvcc --version

.PHONY: all run test clean check
