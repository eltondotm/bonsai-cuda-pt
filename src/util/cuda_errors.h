
#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        std::cerr << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

#define checkCurandErrors(val) check_curand( (val), #val, __FILE__, __LINE__ )

void check_curand(curandStatus_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CuRAND error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}
