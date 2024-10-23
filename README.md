# Capgemini OpenCL tasks
Tasks were implemented and tested on:
 - Windows laptop: GPU: NVIDIA GTX 1050; CPU: Intel Core i7-7700HQ.
 - Linux AWS instance machine: GPU: Nvidia Tesla m60; CPU: Intel Xeon CPU E5-2686.

## Getting Started
As both machines have NVIDIA GPU and installed CUDA toolkit, I've used OpenCL SDK from the CUDA toolkit.

- Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) or OpenCL SDK separately.
- *[optional]* Install [Intel CPU Runtime for OpenCL](https://www.intel.com/content/www/us/en/developer/tools/opencl-cpu-runtime/overview.html) to enable OpenCL on Intel CPU.
- Check OpenCL .lib and headers in the Linux makefile and Windows solution for proper linking.
- Run programs using Make on Linux and Visual Studio 2022 on Windows.
