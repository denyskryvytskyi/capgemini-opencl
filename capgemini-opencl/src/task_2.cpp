/**
 * TASK: Matrix Multiplication Using OpenCL
 * NOTE:
 *  - Implemented and tested on Windows. Laptop: GPU: NVIDIA GTX 1050; CPU: Intel Core i7-7700HQ.
 * RESULTS: (for matrices A(1500; 2000); B(2000; 3000) and result matrix (1500;3000))
 *  - CPU multiplication: ~5500 ms
 *  - GPU multiplication (tiled matrix with shared memory usage kernel matMulTiledKernel): ~89 ms (~x61 faster)
 *      - GPU data preparation (time to allocate GPU buffers and host to device data copy): ~ ms
 *      - Device to host data copy: ~4.5 ms
 * -
 **/

#include "utils.h"

#include <chrono>
#include <iostream>

namespace task_2 {

constexpr int32_t MAT_DIM_N = 1500; // rows of the matrix A
constexpr int32_t MAT_DIM_M = 2000; // cols of the matrix A and rows of the matrix B
constexpr int32_t MAT_DIM_K = 3000; // cols of the matrix B
constexpr int32_t MAT_A_SIZE = MAT_DIM_N * MAT_DIM_M;
constexpr int32_t MAT_B_SIZE = MAT_DIM_M * MAT_DIM_K;
constexpr int32_t MAT_RES_SIZE = MAT_DIM_N * MAT_DIM_K;
constexpr int32_t ALIGNMENT = 16;
constexpr float MAT_A_OFFSET = 0.5f;
constexpr float MAT_B_OFFSET = 1.3f;
constexpr bool PRINT_MAT = false;

// OpenCL specific
const char* const KERNEL_PATH = "kernels/task_2.cl";
const char* const KERNEL_NAME = "matMulTiled";
const char* const PROGRAM_FLAGS = "-cl-mad-enable -cl-fast-relaxed-math";
constexpr size_t TILE_SIZE = 16;
constexpr size_t LOCAL_WORK_SIZE[2] = { TILE_SIZE, TILE_SIZE }; // Amount of work-items per work-group
constexpr size_t GLOBAL_WORK_SIZE[2] = { (MAT_DIM_K + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
                                         (MAT_DIM_N + TILE_SIZE - 1) / TILE_SIZE* TILE_SIZE }; // Total amount of work-items in 2D

// Helpers
void initData(float* pMatA, float* pMatB, float* pMatRes);
void resetRes(float* pMatRes);
void printMat(float* pMat, int32_t rows, int32_t cols);

void matMul(float* pMatA, float* pMatB, float* pMatRes);
void transposeMat(float* pDevMatB, float* pDevMatB_T);
void matMulCl(float* pMatA, float* pMatB, float* pMatRes);

void cleanHost(float* pMatA, float* pMatB, float* pMatRes);
void cleanDevice(cl_mem bufferA, cl_mem bufferB, cl_mem bufferRes, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel);

void run()
{
    float* pMatA = static_cast<float*>(_aligned_malloc(MAT_A_SIZE * sizeof(float), ALIGNMENT));
    if (!pMatA) {
        std::cerr << "Failed to allocate memory for matrix A." << std::endl;
        exit(1);
    }

    float* pMatB = static_cast<float*>(_aligned_malloc(MAT_B_SIZE * sizeof(float), ALIGNMENT));
    if (!pMatB) {
        _aligned_free(pMatA);
        std::cerr << "Failed to allocate memory for matrix B." << std::endl;
        exit(1);
    }

    float* pMatRes = static_cast<float*>(_aligned_malloc(MAT_RES_SIZE * sizeof(float), ALIGNMENT));
    if (!pMatRes) {
        _aligned_free(pMatA);
        _aligned_free(pMatB);
        std::cerr << "Failed to allocate memory for result matrix." << std::endl;
        exit(1);
    }

    initData(pMatA, pMatB, pMatRes);

    if (PRINT_MAT) {
        std::cout << "Mat A:\n";
        printMat(pMatA, MAT_DIM_N, MAT_DIM_M);

        std::cout << "Mat B:\n";
        printMat(pMatB, MAT_DIM_M, MAT_DIM_K);
    }

    resetRes(pMatRes);
    matMul(pMatA, pMatB, pMatRes);

    matMulCl(pMatA, pMatB, pMatRes);

    cleanHost(pMatA, pMatB, pMatRes);
}

void initData(float* pMatA, float* pMatB, float* pMatRes)
{
    for (int i = 0; i < MAT_A_SIZE; ++i) {
        pMatA[i] = static_cast<float>(i) + MAT_A_OFFSET;
    }
    for (int i = 0; i < MAT_B_SIZE; ++i) {
        pMatB[i] = static_cast<float>(i) + MAT_B_OFFSET;
    }
}

void resetRes(float* pMatRes)
{
    memset(pMatRes, 0, MAT_DIM_N * MAT_DIM_K * sizeof(float));
}

void printMat(float* pMat, int32_t rows, int32_t cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << pMat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void matMul(float* pMatA, float* pMatB, float* pMatRes)
{
    std::cout << "===== CPU Matrix Multiplication =====\n";

    // result matrix dim - [N;K]
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < MAT_DIM_N; ++row) {
        for (int col = 0; col < MAT_DIM_K; ++col) {
            for (int i = 0; i < MAT_DIM_M; ++i) {
                pMatRes[row * MAT_DIM_K + col] += pMatA[row * MAT_DIM_M + i] * pMatB[i * MAT_DIM_K + col];
            }
        }
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_MAT) {
        std::cout << "Result matrix:\n";
        printMat(pMatRes, MAT_DIM_N, MAT_DIM_K);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void matMulCl(float* pMatA, float* pMatB, float* pMatRes)
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
        cleanHost(pMatA, pMatB, pMatRes);
        exit(1);
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get device:" << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        exit(1);
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create context:" << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        exit(1);
    }

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating command queue: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 2: Create buffers
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MAT_A_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer A: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MAT_B_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer B: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    bufferRes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * MAT_RES_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating result buffer: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 3: Asynchronously write host data to device memory
    err = clEnqueueWriteBuffer(queue, bufferA, CL_FALSE, 0, sizeof(float) * MAT_A_SIZE, pMatA, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error writing buffer A: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    err = clEnqueueWriteBuffer(queue, bufferB, CL_FALSE, 0, sizeof(float) * MAT_B_SIZE, pMatB, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error writing buffer B: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 4: Create and compile the kernel
    std::string kernelSource = utils::loadKernelSource(KERNEL_PATH);
    const char* testStr = kernelSource.c_str();
    program = clCreateProgramWithSource(context, 1, &testStr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create the program: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
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
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    kernel = clCreateKernel(program, KERNEL_NAME, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create kernel:" << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 5: Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferRes);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &MAT_DIM_N);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &MAT_DIM_M);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &MAT_DIM_K);

    // Step 6: Execute the kernel
    cl_event kernelEvent; // Event for profiling kernel

    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, 0, nullptr, &kernelEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to enqueue the kernel: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 7: Read the results back to the host
    cl_event readEvent; // Event for profiling read from buffer function
    err = clEnqueueReadBuffer(queue, bufferRes, CL_TRUE, 0, sizeof(float) * MAT_RES_SIZE, pMatRes, 1, &kernelEvent, &readEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to read buffer: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatRes);
        cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Step 8: Clean up
    cleanDevice(bufferA, bufferB, bufferRes, context, queue, program, kernel);

    // Step 9: Output the results
    std::cout << "===== GPU Matrix Multiplication =====\n";
    if (PRINT_MAT) {
        std::cout << "Result matrix:\n";
        printMat(pMatRes, MAT_DIM_N, MAT_DIM_K);
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

void cleanHost(float* pMatA, float* pMatB, float* pMatRes)
{
    _aligned_free(pMatA);
    _aligned_free(pMatB);
    _aligned_free(pMatRes);
}

void cleanDevice(cl_mem bufferA, cl_mem bufferB, cl_mem bufferRes, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel)
{
    if (bufferA) {
        clReleaseMemObject(bufferA);
    }
    if (bufferB) {
        clReleaseMemObject(bufferB);
    }
    if (bufferRes) {
        clReleaseMemObject(bufferRes);
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

} // namespace task_2