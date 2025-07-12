
/** 
 * All random functions can be called from host or device functions.
 * However, the random generator must be explicitly initialized for each.
 * 
 * Use init_host() and init_device(width, height) to initialize
 * or reiniitialize the random number generator.
 * 
 * Cleanup with cleanup_host() and cleanup_device()
 * 
 * Host generation assumes future kernel launches are the same size.
 * If the dimensions change, the generator should be reinitialized.
 */

#include "random.h"

#include <iostream>

namespace rng {

///////////////////////////////////////
// Host generator management
///////////////////////////////////////

// Refills buffer and resets current index
__host__ void generate_buffer() {
    checkCurandErrors(curandGenerateUniform(rng, rand_buffer, n));
    i = 0;
}

// Creates generator and allocates buffers, reinitializes on future calls
__host__ void init_host(size_t buffer_size) {
    if (rand_buffer)
        cleanup_host();

    n = buffer_size;

    rand_buffer = (float *)calloc(n, sizeof(float));

    checkCurandErrors(curandCreateGeneratorHost(&rng, CURAND_RNG_PSEUDO_XORWOW));
    checkCurandErrors(curandSetPseudoRandomGeneratorSeed(rng, 725ULL));
    generate_buffer();
}

// Destroys generator and deallocates buffer
__host__ void cleanup_host() {
    free(rand_buffer);
    rand_buffer = nullptr;  // In case of reinitialization
    checkCurandErrors(curandDestroyGenerator(rng));
}


///////////////////////////////////////
// Device generator management
///////////////////////////////////////

__global__ void d_init_device(int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int pixel_idx = j * width + i;
    curand_init(725+pixel_idx, 0, 0, &rand_state[pixel_idx]);
}

// Reinitialize if dimensions change
__host__ void init_device(int width, int height) {
    if (is_initialized)
        cleanup_device();

    checkCudaErrors(cudaMalloc((void **)&h_rand_state, width*height*sizeof(curandState)));
    checkCudaErrors(cudaMemcpyToSymbol(rand_state, &h_rand_state, sizeof(curandState *)));
    checkCudaErrors(cudaMemcpyToSymbol(w, &width, sizeof(int)));

    dim3 threads(8, 8);
    dim3 blocks((width+threads.x-1) / threads.x, (height+threads.y-1) / threads.y);
    d_init_device<<<blocks, threads>>>(width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    is_initialized = true;
}

__host__ void cleanup_device() {
    checkCudaErrors(cudaFree(h_rand_state));
    is_initialized = false;
}


///////////////////////////////////////
// Common random functions
///////////////////////////////////////

// Uniform float in (0, 1]
__host__ __device__ float unit() {
    float val;
    #ifdef __CUDA_ARCH__
    // Recomputes pixel index for convenience, would be better to pass as an argument
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int pixel_idx = j * w + i;  // Assumes bounds checking already done

    val = curand_uniform(&rand_state[pixel_idx]);
    #else
    if (!rand_buffer)
        std::cerr << "Host random number generator not initialized.\n";

    if (i == n) 
        generate_buffer();

    val = rand_buffer[i++];
    #endif
    return val;
}

// Uniformly samples a unit square (0, 1]^2
__host__ __device__ glm::vec2 square() {
    return glm::vec2(unit(), unit());
}

// Uniformly samples a unit disk (Shirley's method for reduced distortion)
__host__ __device__ glm::vec2 disk() {
    glm::vec2 xi = 2.f * square() - glm::vec2(1.f);

    float theta, r;
    if (abs(xi.x) > abs(xi.y)) {
        r = xi.x;
        theta = glm::radians(45.f) * (xi.y / xi.x);
    } else {
        r = xi.y;
        theta = glm::radians(90.f) - glm::radians(45.f) * (xi.x / xi.y);
    }

    return r * glm::vec2(cos(theta), sin(theta));
}

// Uniformly samples a unit hemisphere
__host__ __device__ glm::vec3 hemisphere() {
    glm::vec2 xi = square();

    float theta = acos(xi.x);
    float phi = 2.f * glm::radians(180.f) * xi.y;

    float x = sin(theta) * cos(phi);
    float y = cos(theta);
    float z = sin(theta) * sin(phi);

    return glm::vec3(x, y, z);
}

// Samples hemisphere with cosine weighting
__host__ __device__ glm::vec3 hemisphere_cosine(float &pdf) {
    float eps0 = unit();
    float eps1 = unit();

    float theta = acos(sqrtf(eps0));
    float phi = glm::radians(360.f) * eps1;

    float x = std::sin(theta) * std::cos(phi);
    float y = std::cos(theta);
    float z = std::sin(theta) * std::sin(phi);

    pdf = y / glm::radians(180.f);
    return glm::vec3(x, y, z);
}

}  // namespace rng
