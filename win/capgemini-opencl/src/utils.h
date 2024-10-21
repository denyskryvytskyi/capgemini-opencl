#pragma once

#include "string"
#include <CL/cl.h>

namespace utils {
std::string loadKernelSource(const char* filename);
void profileKernelEvent(cl_event kernelEvent, const std::string& msg);
} // namespace utils
