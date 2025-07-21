
#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>

// Macro to handle errors from CUDA API calls
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

// Macro to handle errors from cuRand API calls
#define checkCurandErrors(val) check_curand( (val), #val, __FILE__, __LINE__ )
void check_curand(curandStatus_t result, char const *const func, const char *const file, int const line);
