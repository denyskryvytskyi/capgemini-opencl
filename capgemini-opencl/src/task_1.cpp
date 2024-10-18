/**
 * TASK: Addition of two vectors using OpenCL
 * NOTE:
 *  - Implemented and tested on Windows. Laptop: GPU: NVIDIA GTX 1050; CPU: Intel Core i7-7700HQ.
 *  - Vectorization (incorporating SIMD instructions to kernels) doesn't improve performance.
 *    Because CUDA is scalar architecture. Therefore, there is no performance benefit from using vector types.
 * RESULTS: (For vectors with the size = 100'000'000)
 *  - CPU addition: ~132 ms
 *  - GPU addition: ~12 ms (x11 faster)
 *  - Device to host memory copy: ~90 ms
 **/

#include "utils.h"

#include <chrono>
#include <iostream>

namespace task_1 {

constexpr size_t VEC_SIZE = 100'000'000;                // Vector size
constexpr size_t GLOBAL_WORK_SIZE = (VEC_SIZE + 3) / 4; // Amount of float4 vectors, handling amount that is not multiple of 4
constexpr size_t LOCAL_WORK_SIZE = 64;                  // Amount of work-items per work-group
constexpr int32_t ALIGNMENT = 16;
constexpr float VEC_A_OFFSET = 0.5f;
constexpr float VEC_B_OFFSET = 1.3f;
constexpr bool PRINT_VEC = false;
const char* const KERNEL_PATH = "kernels/task_1.cl";

// Helpers
void initData(float* pVecA, float* pVecB);
void printVec(float* pVec);
void add(float* pVecA, float* pVecB, float* pVecRes);
void clAdd(float* pVecA, float* pVecB, float* pVecRes);
void cleanHost(float* pVecA, float* pVecB, float* pVecRes);
void cleanDevice(cl_mem bufferA, cl_mem bufferB, cl_mem bufferRes, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel);

void run()
{
    float* pVecA = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecA) {
        std::cout << "Memory allocation failed for vector A." << std::endl;
        exit(1);
    }

    float* pVecB = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecB) {
        _aligned_free(pVecA);
        std::cout << "Memory allocation failed for vector B." << std::endl;
        exit(1);
    }

    float* pVecRes = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecRes) {
        _aligned_free(pVecA);
        _aligned_free(pVecB);
        std::cout << "Memory allocation failed for vector B." << std::endl;
        exit(1);
    }

    initData(pVecA, pVecB);
    if (PRINT_VEC) {
        std::cout << "Vector A: ";
        printVec(pVecA);

        std::cout << "Vector B: ";
        printVec(pVecB);
    }

    add(pVecA, pVecB, pVecRes);

    clAdd(pVecA, pVecB, pVecRes);

    cleanHost(pVecA, pVecB, pVecRes);
}

void initData(float* pVecA, float* pVecB)
{
    for (int i = 0; i < VEC_SIZE; ++i) {
        pVecA[i] = static_cast<float>(i) + VEC_A_OFFSET;
        pVecB[i] = static_cast<float>(i) + VEC_B_OFFSET;
    }
}

void printVec(float* pVec)
{
    for (int i = 0; i < VEC_SIZE; ++i) {
        std::cout << pVec[i] << " ";
    }

    std::cout << std::endl;
}

void add(float* pVecA, float* pVecB, float* pVecRes)
{
    std::cout << "===== CPU Addition =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < VEC_SIZE; ++i) {
        pVecRes[i] = pVecA[i] + pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_VEC) {
        std::cout << "Result: ";
        printVec(pVecRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void clAdd(float* pVecA, float* pVecB, float* pVecRes)
{
    cl_mem bufferA = nullptr;
    cl_mem bufferB = nullptr;
    cl_mem bufferRes = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_int err; // Error code

    // Step 1: Set up OpenCL
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get platform:" << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        exit(1);
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get device:" << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        exit(1);
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create context:" << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        exit(1);
    }

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating command queue: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 2: Create buffers
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float4) * GLOBAL_WORK_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer A: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float4) * GLOBAL_WORK_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer B: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    bufferRes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float4) * GLOBAL_WORK_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating result buffer: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 3: Asynchronously write host data to device memory
    err = clEnqueueWriteBuffer(queue, bufferA, CL_FALSE, 0, sizeof(cl_float4) * GLOBAL_WORK_SIZE, pVecA, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error writing buffer A: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    err = clEnqueueWriteBuffer(queue, bufferB, CL_FALSE, 0, sizeof(cl_float4) * GLOBAL_WORK_SIZE, pVecB, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error writing buffer B: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 4: Create and compile the kernel
    const char* kernelSource = utils::loadKernelSource(KERNEL_PATH).c_str();
    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create the program: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    const char* flags = "-cl-mad-enable -cl-fast-relaxed-math";
    err = clBuildProgram(program, 1, &device, flags, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to build the program: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    kernel = clCreateKernel(program, "add", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create kernel:" << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 5: Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferRes);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &VEC_SIZE);

    // Step 6: Execute the kernel
    cl_event kernelEvent; // Event for profiling kernel

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &GLOBAL_WORK_SIZE, &LOCAL_WORK_SIZE, 0, nullptr, &kernelEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to enqueue the kernel: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 7: Read the results back to the host
    cl_event readEvent; // Event for profiling read from buffer function
    err = clEnqueueReadBuffer(queue, bufferRes, CL_TRUE, 0, sizeof(cl_float4) * GLOBAL_WORK_SIZE, pVecRes, 0, nullptr, &readEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to read buffer: " << err << std::endl;
        cleanHost(pVecA, pVecB, pVecRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 8: Clean up
    cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);

    // Step 9: Output the results
    std::cout << "===== GPU Addition =====\n";
    if (PRINT_VEC) {
        std::cout << "Result: ";
        printVec(pVecRes);
    }

    // Step 10: Profiling the kernel execution time
    cl_ulong startTime, endTime;
    cl_int errStart, errEnd;

    errStart = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, nullptr);
    errEnd = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, nullptr);
    clReleaseEvent(kernelEvent);
    if (errStart != CL_SUCCESS || errEnd != CL_SUCCESS) {
        std::cout << "Error getting profiling info. Start error: " << errStart << ", End error: " << errEnd << std::endl;
    }

    cl_ulong rstartTime, rendTime;
    errStart = clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_START, sizeof(rstartTime), &rstartTime, nullptr);
    errEnd = clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_END, sizeof(rendTime), &rendTime, nullptr);
    clReleaseEvent(readEvent);
    if (errStart != CL_SUCCESS || errEnd != CL_SUCCESS) {
        std::cout << "Error getting profiling info. Start error: " << errStart << ", End error: " << errEnd << std::endl;
    }

    std::cout << "Kernel execution time: " << (endTime - startTime) * 1e-6f << " ms" << std::endl;
    std::cout << "Device to host memory copy time: " << (rendTime - rstartTime) * 1e-6f << " ms" << std::endl;
}

void cleanHost(float* pVecA, float* pVecB, float* pVecRes)
{
    _aligned_free(pVecA);
    _aligned_free(pVecB);
    _aligned_free(pVecRes);
}

void cleanDevice(cl_mem bufferA, cl_mem bufferB, cl_mem bufferRes, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel)
{
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferRes);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

} // namespace task_1
