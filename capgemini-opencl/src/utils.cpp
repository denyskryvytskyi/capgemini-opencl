#include "utils.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace utils {
const char* loadKernelSource(const char* filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();      // Read file into buffer
    return buffer.str().c_str(); // Convert buffer to string
}

} // namespace utils
