#pragma once

#include "string"
#include <CL/cl.h>

namespace utils {
std::string loadKernelSource(const char* filename);
} // namespace utils
