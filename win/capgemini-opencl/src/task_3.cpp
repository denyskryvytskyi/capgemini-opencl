/**
 * TASK:Reduction Operations Using OpenCL
 * NOTE:
 *  - Implemented and tested on Windows. Laptop: GPU: NVIDIA GTX 1050; CPU: Intel Core i7-7700HQ.
 * RESULTS: (Array size = 100'000'050)
 *  - CPU reduction: ~125 ms
 *  - GPU reduction:
 *      - Default: ~5.95 ms (~x21 faster)
 *      - Vectorized: ~3.95 ms (~x31 faster)
 *  - Overhead:
 *      - GPU buffers allocation: ~0.0073 ms
 *      - Device to host memory copy: ~0.28 ms
 *      - Final computation on CPU:
 *          - Default: ~0.47 ms
 *          - Vectorized: ~0.99 ms
 **/

#include "utils.h"

#include <chrono>
#include <iostream>

namespace task_3 {

constexpr int32_t ARR_SIZE = 100'000'050;
constexpr int32_t ALIGNMENT = 16;
constexpr bool PRINT_ARR = false;

// OpenCL specific
const char* const KERNEL_PATH = "kernels/task_3.cl";
const char* const KERNEL_NAME = "reduce";
const char* const PROGRAM_FLAGS = "-cl-mad-enable -cl-fast-relaxed-math -Werror";
constexpr size_t VEC_SIZE = 4;                                                                      // Size of vector for vectorization
constexpr size_t ARR_REMAINDER = ARR_SIZE % VEC_SIZE;                                               // Remainder after array size adjusting for processing multiple of 4 array size
constexpr int32_t ARR_SIZE_VEC = (ARR_SIZE - ARR_REMAINDER) / VEC_SIZE;                             // Adjust array size for vectorized version
constexpr size_t LOCAL_WORK_SIZE = 128;                                                             // Amount of work-items per work-group
constexpr size_t WORK_GROUPS_AMOUNT = ((ARR_SIZE_VEC / 2 + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE); // Amount of work-groups
constexpr size_t GLOBAL_WORK_SIZE = WORK_GROUPS_AMOUNT * LOCAL_WORK_SIZE;                           // Total amount of work-items

// Helpers
void initData(float* pArr);
void printArr(float* pArr);

void reduce(float* pArr);
void reduceCl(float* pArr, float* pArrOut);

void cleanHost(float* pArr, float* pArrOut);
void cleanDevice(cl_mem buffer, cl_mem bufferOut, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel);

void run()
{
    float* pArr = static_cast<float*>(_aligned_malloc(ARR_SIZE * sizeof(float), ALIGNMENT));
    if (!pArr) {
        std::cerr << "Failed to allocate memory for array." << std::endl;
        exit(1);
    }

    initData(pArr);

    if (PRINT_ARR) {
        std::cout << "Array: ";
        printArr(pArr);
    }

    reduce(pArr);

    // Partial reduction output array after parallel execution on GPU
    float* pArrOut = static_cast<float*>(_aligned_malloc(WORK_GROUPS_AMOUNT * sizeof(float) * 4, ALIGNMENT));
    if (!pArrOut) {
        std::cerr << "Memory allocation failed for array." << std::endl;
        _aligned_free(pArr);
        exit(1);
    }

    std::cout << "===== GPU Matrix Multiplication =====\n";

    reduceCl(pArr, pArrOut);

    cleanHost(pArr, pArrOut);
}

void initData(float* pArr)
{
    for (int i = 0; i < ARR_SIZE; ++i) {
        pArr[i] = static_cast<float>(i);
    }
}

void printArr(float* pVec)
{
    for (int i = 0; i < ARR_SIZE; ++i) {
        std::cout << pVec[i] << " ";
    }

    std::cout << std::endl;
}

void reduce(float* pArr)
{
    std::cout << "===== CPU Reduction =====\n";

    double sum = 0.0f; // double to avoid float precision issue

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ARR_SIZE; ++i) {
        sum += pArr[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    std::cout << "Reduction (sum): " << sum << std::endl;

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void reduceCl(float* pArr, float* pArrOut)
{
    cl_mem buffer = nullptr;
    cl_mem bufferOut = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_kernel kernelTranspose = nullptr;

    cl_int err; // Error code

    // Set up OpenCL (platform -> device -> context -> queue)
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get platform:" << err << std::endl;
        cleanHost(pArr, pArrOut);
        exit(1);
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get device:" << err << std::endl;
        cleanHost(pArr, pArrOut);
        exit(1);
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create context:" << err << std::endl;
        cleanHost(pArr, pArrOut);
        exit(1);
    }

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating command queue: " << err << std::endl;
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }

    // Create buffers
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float4) * ARR_SIZE_VEC, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer: " << err << std::endl;
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }
    bufferOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float4) * WORK_GROUPS_AMOUNT, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating output buffer: " << err << std::endl;
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    // Asynchronously write host data to device memory
    err = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, sizeof(cl_float4) * ARR_SIZE_VEC, pArr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error writing buffer: " << err << std::endl;
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }

    // Create and compile the program
    std::string kernelSource = utils::loadKernelSource(KERNEL_PATH);
    const char* testStr = kernelSource.c_str();
    program = clCreateProgramWithSource(context, 1, &testStr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create the program: " << err << std::endl;
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }

    err = clBuildProgram(program, 1, &device, PROGRAM_FLAGS, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to build the program: " << err << std::endl;

        {
            // logs of build
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char* log = new char[log_size];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            std::cerr << "Build Log:\n"
                      << log << std::endl;
            delete[] log;
        }
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }

    // Create kernel
    kernel = clCreateKernel(program, KERNEL_NAME, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create kernel:" << err << std::endl;
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to set kernel argument 0: " << err << std::endl;
    }
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferOut);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to set kernel argument 1: " << err << std::endl;
    }
    err = clSetKernelArg(kernel, 2, sizeof(cl_int), &ARR_SIZE_VEC);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to set kernel argument 2: " << err << std::endl;
    }
    size_t localMemSize = LOCAL_WORK_SIZE * sizeof(cl_float4);
    err = clSetKernelArg(kernel, 3, localMemSize, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to set kernel argument 3: " << err << std::endl;
    }

    // Execute the kernel
    cl_event kernelEvent; // Event for profiling kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &GLOBAL_WORK_SIZE, &LOCAL_WORK_SIZE, 0, nullptr, &kernelEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to enqueue the kernel: " << err << std::endl;
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }

    // Read the results back to the host
    cl_event readEvent; // Event for profiling read from buffer function
    err = clEnqueueReadBuffer(queue, bufferOut, CL_TRUE, 0, sizeof(cl_float4) * WORK_GROUPS_AMOUNT, pArrOut, 1, &kernelEvent, &readEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to read buffer: " << err << std::endl;
        cleanHost(pArr, pArrOut);
        cleanDevice(buffer, bufferOut, context, queue, program, kernel);
        exit(1);
    }

    // Final result computation
    const auto fstartTimePoint = std::chrono::high_resolution_clock::now();
    double sum = 0.0f; // double to avoid float precision issue
    for (int i = 0; i < WORK_GROUPS_AMOUNT * VEC_SIZE; ++i) {
        sum += pArrOut[i];
    }
    // Remainder
    for (int i = ARR_SIZE - ARR_REMAINDER; i < ARR_SIZE; ++i) {
        sum += pArr[i];
    }
    const auto fendTimePoint = std::chrono::high_resolution_clock::now();

    // Clean up
    cleanDevice(buffer, bufferOut, context, queue, program, kernel);

    // Output the results
    std::cout << "Reduction (sum): " << sum << std::endl;

    // Profiling the kernel execution time
    const auto duration = std::chrono::duration<double, std::milli>(endTimePoint - startTimePoint);
    std::cout << "Buffers allocation time: " << duration.count() << " ms.\n";

    utils::profileKernelEvent(kernelEvent, "Kernel execution time: ");
    utils::profileKernelEvent(readEvent, "Device to host memory copy time: ");

    const auto fduration = std::chrono::duration<double, std::milli>(fendTimePoint - fstartTimePoint);
    std::cout << "Final computation on CPU: " << fduration.count() << " ms.\n";
}

void cleanHost(float* pArr, float* pArrOut)
{
    _aligned_free(pArr);
    _aligned_free(pArrOut);
}

void cleanDevice(cl_mem buffer, cl_mem bufferOut, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel)
{
    if (buffer) {
        clReleaseMemObject(buffer);
    }
    if (bufferOut) {
        clReleaseMemObject(bufferOut);
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

} // namespace task_3
