#pragma once

#include <CL/cl.h>

namespace utils {
    const char* loadKernelSource(const char* filename);
    void checkError(cl_int err, const char* operation);
}
