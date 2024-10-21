# Makefile for compiling task_1.cpp with OpenCL support

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -I/usr/local/cuda/include -I../utils -L/usr/local/cuda/lib64 -DCL_HPP_TARGET_OPENCL_VERSION=300

# Linker flags
LDFLAGS = -lOpenCL

# Source files
SRCS = task_1.cpp

# Output binary
TARGET = task_1

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Clean up the build
clean:
	rm -f $(TARGET)

.PHONY: all clean