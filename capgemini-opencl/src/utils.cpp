#include "utils.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace utils {
std::string loadKernelSource(const char* filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }

    std::stringstream buffer;
    buffer << file.rdbuf(); // Read file into buffer
    return buffer.str();    // Convert buffer to string
}

void profileKernelEvent(cl_event kernelEvent, const std::string& msg)
{
    cl_ulong startTime, endTime;
    cl_int errStart, errEnd;
    errStart = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, nullptr);
    errEnd = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, nullptr);

    clReleaseEvent(kernelEvent);
    if (errStart != CL_SUCCESS || errEnd != CL_SUCCESS) {
        std::cout << "Error getting profiling info. Start error: " << errStart << ", End error: " << errEnd << std::endl;
    }

    std::cout << msg << (endTime - startTime) * 1e-6f << " ms" << std::endl;
}

} // namespace utils
