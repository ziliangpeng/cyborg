# Makefile for CUDA programs

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O2 -arch=sm_70

# Target executable
TARGET = vector_add

# Source files
SOURCES = vector_add.cu

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Check if CUDA is available
check:
	@echo "Checking CUDA installation..."
	@which nvcc || echo "ERROR: nvcc not found. Please install CUDA toolkit."
	@nvcc --version || true

.PHONY: all run clean check
