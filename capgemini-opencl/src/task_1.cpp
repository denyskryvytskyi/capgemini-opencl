#include "utils.h"

#include <iostream>
#include <chrono>

namespace task_1{

constexpr int32_t VEC_SIZE = 100'000'000;
constexpr int32_t ALIGNMENT = 16;
constexpr float VEC_A_OFFSET = 0.5f;
constexpr float VEC_B_OFFSET = 1.3f;

void initData(float* pVecA, float* pVecB);
void add(float* pVecA, float* pVecB, float* pVecRes);

int run() {
    float* pVecA = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecA) {
        std::cout << "Memory allocation failed for vector A." << std::endl;
        return 1;
    }

    float* pVecB = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecB) {
        _aligned_free(pVecA);
        std::cout << "Memory allocation failed for vector B." << std::endl;
        return 1;
    }

    float* pVecRes = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecRes) {
        _aligned_free(pVecA);
        _aligned_free(pVecB);
        std::cout << "Memory allocation failed for vector B." << std::endl;
        return 1;
    }

    initData(pVecA, pVecB);
    add(pVecA, pVecB, pVecRes);

    // Step 1: Set up OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

    cl_int err;
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating command queue: " << err << std::endl;
        return 1;
    }

    // Step 2: Create buffers
    cl_int ciErrBuffer;
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VEC_SIZE, nullptr, &ciErrBuffer);
    if (ciErrBuffer != CL_SUCCESS) {
        std::cout << "Error creating buffer A." << ciErrBuffer << std::endl;
    }
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * VEC_SIZE, nullptr, &ciErrBuffer);
    if (ciErrBuffer != CL_SUCCESS) {
        std::cout << "Error creating buffer A." << ciErrBuffer << std::endl;
    }
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * VEC_SIZE, nullptr, &ciErrBuffer);
    if (ciErrBuffer != CL_SUCCESS) {
        std::cout << "Error creating buffer A." << ciErrBuffer << std::endl;
    }

    // Step : Asynchronously write data to device
    
    ciErrBuffer = clEnqueueWriteBuffer(queue, bufferA, CL_FALSE, 0, sizeof(float) * VEC_SIZE, pVecA, 0, nullptr, nullptr);
    if (ciErrBuffer != CL_SUCCESS) {
        std::cout << "Error writing buffer A." << ciErrBuffer << std::endl;
    }
    ciErrBuffer = clEnqueueWriteBuffer(queue, bufferB, CL_FALSE, 0, sizeof(float) * VEC_SIZE, pVecB, 0, nullptr, nullptr);
    if (ciErrBuffer != CL_SUCCESS) {
        std::cout << "Error writing buffer A." << ciErrBuffer << std::endl;
    }

    // Step 3: Create and compile the kernel
    const char* kernelSource = utils::loadKernelSource("kernels/task_1.cl");
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);

    // Step 4: Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &VEC_SIZE);

    // Step 5: Execute the kernel
    cl_event kernelEvent; // Event for profiling kernel
    size_t globalWorkSize = VEC_SIZE;
    size_t localWorkSize = 256;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, &kernelEvent);

    // Step 6: Read the results back to the host
    cl_event readEvent; // Event for profiling read from buffer
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * VEC_SIZE, pVecRes, 0, nullptr, &readEvent);
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

   
    cl_ulong startTime, endTime;
    cl_int err_start, err_end;  // To check for errors
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, nullptr);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, nullptr);

    err_start = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, nullptr);
    err_end = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, nullptr);

    if (err_start != CL_SUCCESS || err_end != CL_SUCCESS) {
        std::cout << "Error getting profiling info. Start error: " << err_start << ", End error: " << err_end << std::endl;
        // Handle the error appropriately
    }

    cl_command_queue_properties actual_props;
    clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(actual_props), &actual_props, nullptr);
    if (!(actual_props & CL_QUEUE_PROFILING_ENABLE)) {
        std::cout << "Profiling is not enabled!" << std::endl;
    }

    cl_ulong rstartTime, rendTime;
    clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_START, sizeof(rstartTime), &rstartTime, nullptr);
    clGetEventProfilingInfo(readEvent, CL_PROFILING_COMMAND_END, sizeof(rendTime), &rendTime, nullptr);


    // Step 7: Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Output the results
    for (int i = VEC_SIZE - 10; i < VEC_SIZE; ++i) { // Print first 10 results
        std::cout << pVecRes[i] << " "; // Should print 3.0 for each element
    }
    std::cout << std::endl;

    // Step 9: Profiling the kernel execution time
   
    std::cout << "Kernel execution time: " << (endTime - startTime) * 1e-6f << " ms" << std::endl; // Convert to milliseconds
    std::cout << "Kernel execution time (read buffer): " << (rendTime - rstartTime) * 1e-6f << " ms" << std::endl; // Convert to milliseconds

    std::cout << "===== GPU Addition =====\n";
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "GPU exec time: " << duration.count() << " ms.\n";

    // Release the event
    clReleaseEvent(kernelEvent);

    return 0;
}

void initData(float* pVecA, float* pVecB)
{
    for (int i = 0, n = VEC_SIZE; i < VEC_SIZE; ++i, --n) {
        pVecA[i] = static_cast<float>(n) + VEC_A_OFFSET;
        pVecB[i] = static_cast<float>(n) + VEC_B_OFFSET;
    }
}

void add(float* pVecA, float* pVecB, float* pVecRes)
{
    std::cout << "===== CPU Addition =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < VEC_SIZE; ++i) {
        pVecRes[i] = pVecA[i] + pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}


}