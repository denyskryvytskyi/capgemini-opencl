/**
 * TASK: Matrix Multiplication Using OpenCL
 * NOTE:
 *  - Implemented and tested on Windows and Linux.
 *      - Windows (laptop): GPU: NVIDIA GTX 1050; CPU: Intel Core i7-7700HQ.
 *      - Linux (this machine): GPU: Nvidia Tesla m60; CPU: Intel Xeon CPU E5-2686.
 *  - OpenCL version based on tiled matrix with shared memory and transposed matrix B logic.
 * RESULTS: (for matrices A(1500; 2000); B(2000; 3000) and result matrix (1500;3000))
 *  - Windows:
 *      - CPU loop: ~5000 ms
 *      - GPU device (NVIDIA GTX 1050) kernel : ~81 ms (~x61 faster)
 *          - Overhead:
 *              - GPU buffers allocation: ~0.02 ms
 *              - Matrix B transposition: ~1.9 ms
 *              - Device to host result matrix copy: ~4.1 ms
 *      - GPU device (Intel HD Graphics) kernel : ~808 ms (~x6.2 faster)
 *          - Overhead:
 *              - GPU buffers allocation: ~0.33 ms
 *              - Matrix B transposition: ~3.9 ms
 *              - Device to host result matrix copy: ~2.3 ms
 *      - CPU device (Intel Core i7-7700HQ) kernel : ~290 ms (~x17 faster)
 *          - Overhead:
 *              - GPU buffers allocation: ~0.05 ms
 *              - Matrix B transposition: ~12.7 ms
 *              - Device to host result matrix copy: ~1.95 ms
*  - Linux:
 *      - CPU loop: ~31160 ms
 *      - GPU device (NVIDIA) kernel: ~50.5 ms (~x1920 faster)
 *      - Overhead:
 *          - GPU buffers allocation: ~0.012 ms
 *          - Matrix B transposition: ~0.95 ms
 *          - Device to host result matrix copy: ~2.02 ms
 **/

#include "utils.h" // load and profile kernel functions

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

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
const char* const KERNEL_PATH = "task_2.cl";
const char* const KERNEL_NAME = "matMulTiled";
const char* const TRANPOSE_KERNEL_NAME = "transpose";
const char* const PROGRAM_FLAGS = "-cl-mad-enable -cl-fast-relaxed-math";
constexpr size_t TILE_SIZE = 16;
constexpr size_t TILES_AMOUNT = (MAT_DIM_M + TILE_SIZE - 1) / TILE_SIZE; // Amount of tiles to cover result matrix
constexpr size_t LOCAL_WORK_SIZE[2] = { TILE_SIZE, TILE_SIZE };          // Amount of work-items per work-group
constexpr size_t GLOBAL_WORK_SIZE[2] = { ((MAT_DIM_K + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE,
                                         ((MAT_DIM_N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE }; // Total amount of work-items in 2D
constexpr size_t GLOBAL_WORK_SIZE_TRANSPOSE[2] = { MAT_DIM_M, MAT_DIM_K };                        // Total amount of work-items in 2D for transpose kernel

// Helpers
void initData(float* pMatA, float* pMatB, float* pMatRes);
void printMat(float* pMat, int32_t rows, int32_t cols);

void matMul(float* pMatA, float* pMatB, float* pMatRes);
void matMulCl(float* pMatA, float* pMatB, float* pMatB_T, float* pMatRes, cl_device_id device);

void cleanHost(float* pMatA, float* pMatB, float* pMatB_T, float* pMatRes);
void cleanDevice(cl_mem bufferA, cl_mem bufferB, cl_mem bufferB_T, cl_mem bufferRes, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel);

int main()
{
    float* pMatA = nullptr;
    float* pMatB = nullptr;
    float* pMatB_T = nullptr;
    float* pMatRes = nullptr;

    if (posix_memalign(reinterpret_cast<void**>(&pMatA), ALIGNMENT, MAT_A_SIZE * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate memory for matrix A." << std::endl;
        exit(1);
    }

    if (posix_memalign(reinterpret_cast<void**>(&pMatB), ALIGNMENT, MAT_B_SIZE * sizeof(float)) != 0) {
        free(pMatA);
        std::cerr << "Failed to allocate memory for matrix B." << std::endl;
        exit(1);
    }

    if (posix_memalign(reinterpret_cast<void**>(&pMatB_T), ALIGNMENT, MAT_B_SIZE * sizeof(float)) != 0) {
        free(pMatA);
        free(pMatB);
        std::cerr << "Failed to allocate memory for transpose matrix B ." << std::endl;
        exit(1);
    }

    if (posix_memalign(reinterpret_cast<void**>(&pMatRes), ALIGNMENT, MAT_RES_SIZE * sizeof(float)) != 0) {
        free(pMatA);
        free(pMatB);
        free(pMatB_T);
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

    matMul(pMatA, pMatB, pMatRes);

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

        matMulCl(pMatA, pMatB, pMatB_T, pMatRes, device);

        std::cout << std::endl;
    }

    cleanHost(pMatA, pMatB, pMatB_T, pMatRes);

    return 0;
}

void initData(float* pMatA, float* pMatB, float* pMatRes)
{
    for (int i = 0; i < MAT_A_SIZE; ++i) {
        pMatA[i] = static_cast<float>(i) + MAT_A_OFFSET;
    }
    for (int i = 0; i < MAT_B_SIZE; ++i) {
        pMatB[i] = static_cast<float>(i) + MAT_B_OFFSET;
    }

    memset(pMatRes, 0, MAT_RES_SIZE * sizeof(float));
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

void matMulCl(float* pMatA, float* pMatB, float* pMatB_T, float* pMatRes, cl_device_id device)
{
    cl_mem bufferA = nullptr;
    cl_mem bufferB = nullptr;
    cl_mem bufferB_T = nullptr;
    cl_mem bufferRes = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_kernel kernelTranspose = nullptr;

    cl_int err; // Error code

    // Set up OpenCL (context -> queue)
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create context:" << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        exit(1);
    }

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating command queue: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Create buffers
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MAT_A_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer A: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MAT_B_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer B: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    bufferB_T = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * MAT_B_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating buffer B: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    bufferRes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * MAT_RES_SIZE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating result buffer: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    // Asynchronously write host data to device memory
    err = clEnqueueWriteBuffer(queue, bufferA, CL_FALSE, 0, sizeof(float) * MAT_A_SIZE, pMatA, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error writing buffer A: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }
    err = clEnqueueWriteBuffer(queue, bufferB, CL_FALSE, 0, sizeof(float) * MAT_B_SIZE, pMatB, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error writing buffer B: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Create and compile the program
    std::string kernelSource = loadKernelSource(KERNEL_PATH);
    const char* kernelSourceCstr = kernelSource.c_str();
    program = clCreateProgramWithSource(context, 1, &kernelSourceCstr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create the program: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    err = clBuildProgram(program, 1, &device, PROGRAM_FLAGS, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to build the program: " << err << std::endl;

        // logs of build
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        char* log = new char[logSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        std::cerr << "Build Log:\n" << log << std::endl;
        delete[] log;
        
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // ========================= Matrix B transposition =============================

    // Create tranpose kernel
    kernelTranspose = clCreateKernel(program, TRANPOSE_KERNEL_NAME, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create transpose kernel:" << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Set kernel arguments
    clSetKernelArg(kernelTranspose, 0, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernelTranspose, 1, sizeof(cl_mem), &bufferB_T);
    clSetKernelArg(kernelTranspose, 2, sizeof(cl_int), &MAT_DIM_M);
    clSetKernelArg(kernelTranspose, 3, sizeof(cl_int), &MAT_DIM_K);

    // Execute the kernel
    cl_event kernelTransposeEvent; // Event for profiling kernel
    err = clEnqueueNDRangeKernel(queue, kernelTranspose, 2, nullptr, GLOBAL_WORK_SIZE_TRANSPOSE, nullptr, 0, nullptr, &kernelTransposeEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to enqueue the transpose kernel: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // ===============================================================================

    // Create main kernel
    kernel = clCreateKernel(program, KERNEL_NAME, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create kernel:" << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB_T);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferRes);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &MAT_DIM_N);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &MAT_DIM_M);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &MAT_DIM_K);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &TILES_AMOUNT);

    // Execute the kernel
    cl_event kernelEvent; // Event for profiling kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, 0, nullptr, &kernelEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to enqueue the kernel: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Read the results back to the host
    cl_event readEvent; // Event for profiling read from buffer function
    err = clEnqueueReadBuffer(queue, bufferRes, CL_TRUE, 0, sizeof(float) * MAT_RES_SIZE, pMatRes, 1, &kernelEvent, &readEvent);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to read buffer: " << err << std::endl;
        cleanHost(pMatA, pMatB, pMatB_T, pMatRes);
        cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);
        exit(1);
    }

    // Clean up
    cleanDevice(bufferA, bufferB, bufferB_T, bufferRes, context, queue, program, kernel);

    // Output the results
    if (PRINT_MAT) {
        std::cout << "Result matrix:\n";
        printMat(pMatRes, MAT_DIM_N, MAT_DIM_K);
    }

    // Profiling the kernel execution time
    const auto duration = std::chrono::duration<double, std::milli>(endTimePoint - startTimePoint);
    std::cout << "Buffers allocation time: " << duration.count() << " ms.\n";

    profileKernelEvent(kernelTransposeEvent, "Transpose kernel execution time: ");
    profileKernelEvent(kernelEvent, "Kernel execution time: ");
    profileKernelEvent(readEvent, "Device to host memory copy time: ");
}

void cleanHost(float* pMatA, float* pMatB, float* pMatB_T, float* pMatRes)
{
    free(pMatA);
    free(pMatB);
    free(pMatB_T);
    free(pMatRes);
}

void cleanDevice(cl_mem bufferA, cl_mem bufferB, cl_mem bufferB_T, cl_mem bufferRes, cl_context context, cl_command_queue queue, cl_program program, cl_kernel kernel)
{
    if (bufferA) {
        clReleaseMemObject(bufferA);
    }
    if (bufferB) {
        clReleaseMemObject(bufferB);
    }
    if (bufferB_T) {
        clReleaseMemObject(bufferB_T);
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