# Makefile for CUDA programs

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O2 -arch=sm_90

# Targets
TARGETS = vector_add mem_benchmark

# Default target - build all
all: $(TARGETS)

# Build vector_add from multiple source files
vector_add: vector_add.cu vector_kernels.cu vector_init.cu cuda_utils.h vector_kernels.h vector_init.h
	$(NVCC) $(NVCC_FLAGS) vector_add.cu vector_kernels.cu vector_init.cu -o vector_add

# Build mem_benchmark
mem_benchmark: mem_benchmark.cu cuda_utils.h
	$(NVCC) $(NVCC_FLAGS) mem_benchmark.cu -o mem_benchmark

# Run vector_add
run: vector_add
	./vector_add

# Test all binaries
test: all
	@echo "=========================================="
	@echo "Running verification tests..."
	@echo "=========================================="
	@echo "Testing vector_add..."
	@./vector_add -n 1000 -v > /dev/null && echo "  ✓ 1K elements"
	@./vector_add -n 1000000 -v > /dev/null && echo "  ✓ 1M elements"
	@./vector_add -n 10000000 -v > /dev/null && echo "  ✓ 10M elements"
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
	@which nvcc || echo "ERROR: nvcc not found. Please install CUDA toolkit."
	@nvcc --version || true

.PHONY: all run test clean check
