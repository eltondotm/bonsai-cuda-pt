# Bonsai Cuda Pathtracer

Simple GPU pathtracer for investigating BVH traversal.

## Getting Started

### Dependencies

* CUDA Toolkit (<link>https://developer.nvidia.com/cuda-toolkit</link>)
* CMake 3.17 or later

### Building

Currently, the GPU architecture NVCC generates for is hardcoded.
You may need to change it in CMakeLists.txt for your architecture.

Included CMake presets are:
* x64-debug
* x64-release
* x86-debug
* x86-release

### Executing program

* Can be run with no arguments to render a (noisy) Cornell Box.
* See argument options with -h (or any invalid arguments).
```
.\pt.exe
.\pt.exe -w 1280 -h 720 -s 4096 -f assets/cbox.mini -o image.png
