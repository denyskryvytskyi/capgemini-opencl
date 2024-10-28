#pragma once

#include <CL/cl.h>
#include <string>

namespace utils {
std::string loadKernelSource(const char* filename);
void profileKernelEvent(cl_event kernelEvent, const std::string& msg);
} // namespace utils
