/**
 * TASK: Sorting with OpenCL
 * NOTE:
 *  - Implemented and tested on Windows. Laptop: GPU: NVIDIA GTX 1050; CPU: Intel Core i7-7700HQ.
 * RESULTS: (Bitonic Sort for array size = 134'217'728)
 *  - CPU loop sort: ~4000 ms
 *  - GPU device (NVIDIA) kernel: ~1560 ms
 *      - Overhead:
 *          - Buffers allocation: ~134 ms
 *          - Device to host memory copy: ~121 ms
 *  - CPU device (Intel Core i7) kernel: ~5000 ms
 *      - Overhead:
 *          - Buffers allocation: ~0.3 ms
 *          - Device to host memory copy: ~66 ms
 **/

#include "utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

namespace task_4 {

constexpr int32_t ARR_SIZE = 1 << 27;
constexpr int32_t ARR_MIN_VAL = -100;
constexpr int32_t ARR_MAX_VAL = 100;
constexpr int32_t ALIGNMENT = 16;
constexpr bool PRINT_ARR = false;

// OpenCL specific
const char* const KERNEL_PATH = "kernels/task_4.cl";
const char* const KERNEL_NAME = "sort";
const char* const PROGRAM_FLAGS = "-cl-mad-enable -cl-fast-relaxed-math -Werror";
constexpr size_t LOCAL_WORK_SIZE = 128;
constexpr size_t GLOBAL_WORK_SIZE = (ARR_SIZE / 2 + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE * LOCAL_WORK_SIZE;

// Helpers
void initData(int32_t* pArr);
void printArr(int32_t* pArr);

void sort(int32_t* pArr);
void sortCl(int32_t* pArr, cl_device_id device);

void cleanDevice(cl_mem buffer, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel);

void run()
{
    int32_t* pArr = static_cast<int32_t*>(_aligned_malloc(ARR_SIZE * sizeof(int32_t), ALIGNMENT));
    if (!pArr) {
        std::cerr << "Memory allocation failed for array." << std::endl;
        exit(1);
    }

    initData(pArr);

    if (PRINT_ARR) {
        std::cout << "Array: ";
        printArr(pArr);
    }

    sort(pArr);

    // ===== OpenCL version =====
    std::cout << "\n===== OpenCL version =====\n";

    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms); // Get number of platforms
    if (numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        exit(1);
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    std::cout << "Supported platforms found: " << numPlatforms << std::endl;

    cl_int err;
    for (const auto& platform : platforms) {
        char platformName[128];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platformName, nullptr);
        std::cout << "===== Platform: " << platformName << " ======\n";

        cl_device_id device = nullptr;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to get GPU device: " << err << std::endl;

            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
            if (err != CL_SUCCESS) {
                std::cout << "Failed to get CPU device: " << err << std::endl;
                continue;
            }
        }

        sortCl(pArr, device);

        std::cout << std::endl;
    }

    _aligned_free(pArr);
}

void initData(int32_t* pArr)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(ARR_MIN_VAL, ARR_MAX_VAL); // range [min, max]

    for (int i = 0; i < ARR_SIZE; ++i) {
        pArr[i] = distr(gen);
    }
}

void printArr(int32_t* pVec)
{
    for (int i = 0; i < ARR_SIZE; ++i) {
        std::cout << pVec[i] << " ";
    }

    std::cout << std::endl;
}

void sort(int32_t* pArr)
{
    std::cout << "===== CPU Sorting =====\n";

    // Allocate copy just to enable the same array be sorted using opencl based function
    int32_t* pArrCopy = static_cast<int32_t*>(_aligned_malloc(ARR_SIZE * sizeof(int32_t), ALIGNMENT));

    if (!pArrCopy) {
        std::cerr << "Memory allocation failed for array copy." << std::endl;
        exit(1);
    }

    memcpy(pArrCopy, pArr, ARR_SIZE * sizeof(int32_t));

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    std::sort(pArrCopy, pArrCopy + ARR_SIZE);
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_ARR) {
        std::cout << "Sorted array: ";
        printArr(pArrCopy);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void sortCl(int32_t* pArr, cl_device_id device)
{
    cl_mem buffer = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_int err; // Error code

    // Set up OpenCL (context -> queue)

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create context:" << err << std::endl;
        _aligned_free(pArr);
        exit(1);
    }

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating command queue: " << err << std::endl;
        _aligned_free(pArr);
        cleanDevice(buffer, context, queue, program, kernel);
        exit(1);
    }

    // Create buffers
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t) * ARR_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer: " << err << std::endl;
        _aligned_free(pArr);
        cleanDevice(buffer, context, queue, program, kernel);
        exit(1);
    }

    // Asynchronously write host data to device memory
    err = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, sizeof(int32_t) * ARR_SIZE, pArr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error writing buffer: " << err << std::endl;
        _aligned_free(pArr);
        cleanDevice(buffer, context, queue, program, kernel);
        exit(1);
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    // Create and compile the program
    std::string kernelSource = utils::loadKernelSource(KERNEL_PATH);
    const char* kernelSourceCstr = kernelSource.c_str();
    program = clCreateProgramWithSource(context, 1, &kernelSourceCstr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create the program: " << err << std::endl;
        _aligned_free(pArr);
        cleanDevice(buffer, context, queue, program, kernel);
        exit(1);
    }

    err = clBuildProgram(program, 1, &device, PROGRAM_FLAGS, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to build the program: " << err << std::endl;

        // Logs of build
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        char* log = new char[log_size];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
        std::cerr << "Build Log:\n"
                  << log << std::endl;
        delete[] log;

        _aligned_free(pArr);
        cleanDevice(buffer, context, queue, program, kernel);
        exit(1);
    }

    // Create kernel
    kernel = clCreateKernel(program, KERNEL_NAME, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create kernel:" << err << std::endl;
        _aligned_free(pArr);
        cleanDevice(buffer, context, queue, program, kernel);
        exit(1);
    }

    std::vector<cl_event> kernelEvents;

    // Bitonic Sort
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer); // Set data
    for (int stage = 2; stage <= ARR_SIZE; stage *= 2) {
        for (int stride = stage >> 1; stride > 0; stride >>= 1) {
            // Set kernel arguments
            clSetKernelArg(kernel, 1, sizeof(int), &ARR_SIZE); // Total length
            clSetKernelArg(kernel, 2, sizeof(int), &stride);   // Stride between compared elements
            clSetKernelArg(kernel, 3, sizeof(int), &stage);    // Current stage

            // Launch kernel
            cl_event kernelEvent;
            err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &GLOBAL_WORK_SIZE, &LOCAL_WORK_SIZE, 0, nullptr, &kernelEvent);
            if (err != CL_SUCCESS) {
                std::cout << "Failed to enqueu kernel:" << err << std::endl;
                _aligned_free(pArr);
                cleanDevice(buffer, context, queue, program, kernel);
                exit(1);
            }
            kernelEvents.emplace_back(kernelEvent);
        }
    }

    cl_event readEvent; // Event for profiling read from buffer function
    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(int32_t) * ARR_SIZE, pArr, 0, nullptr, &readEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to read buffer: " << err << std::endl;
        _aligned_free(pArr);
        cleanDevice(buffer, context, queue, program, kernel);
        exit(1);
    }

    // Clean up
    cleanDevice(buffer, context, queue, program, kernel);

    // Output the results
    if (PRINT_ARR) {
        std::cout << "Sorted array: ";
        printArr(pArr);
    }

    // Profiling the kernel execution time
    const auto duration = std::chrono::duration<double, std::milli>(endTimePoint - startTimePoint);
    std::cout << "Buffers allocation time: " << duration.count() << " ms.\n";

    // Get profiling info of the kernel
    cl_ulong startTime, endTime;
    double totalExecutionTime = 0.0;
    for (auto& kernelEvent : kernelEvents) {
        clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, nullptr);
        clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, nullptr);
        totalExecutionTime += (endTime - startTime) * 1e-6; // Convert to milliseconds
        clReleaseEvent(kernelEvent);
    }

    std::cout << "Kernel execution time:  " << totalExecutionTime << " ms.\n";
    utils::profileKernelEvent(readEvent, "Device to host memory copy time: ");
}

void cleanDevice(cl_mem buffer, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel)
{
    if (buffer) {
        clReleaseMemObject(buffer);
    }
    if (kernel) {
        clReleaseKernel(kernel);
    }
    if (program) {
        clReleaseProgram(program);
    }
    if (queue) {
        clReleaseCommandQueue(queue);
    }
    if (context) {
        clReleaseContext(context);
    }
}

} // namespace task_4
