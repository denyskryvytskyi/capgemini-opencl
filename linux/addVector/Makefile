# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -I/usr/local/cuda/include -I../utils -L/usr/local/cuda/lib64 -DCL_HPP_TARGET_OPENCL_VERSION=300

# Linker flags
LDFLAGS = -lOpenCL -O3

# Source files
SRCS = addVector.cpp

# Output binary
TARGET = addVector

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Clean up the build
clean:
	rm -f $(TARGET)

.PHONY: all clean