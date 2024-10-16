#include "utils.h"

#include <fstream>
#include <sstream>
#include <iostream>

namespace utils {
    const char* loadKernelSource(const char* filename)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "Error: Failed to open kernel file: " << filename << std::endl;
            exit(1);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();  // Read file into buffer
        return buffer.str().c_str();      // Convert buffer to string
    }

    void checkError(cl_int err, const char* operation)
    {
        if (err != CL_SUCCESS) {
            std::cout << "Error during operation '" << operation << "': " << err << std::endl;
            exit(1);
        }
    }
}
