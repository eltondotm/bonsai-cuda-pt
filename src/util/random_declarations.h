
#pragma once

#include <cuda_runtime.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

// Forward declarations of functions from random.h to avoid multiple definitions
namespace rng {
    __host__ void generate_buffer();
    __host__ void cleanup_host();
    __host__ void init_host(size_t buffer_size = 1024);

    __host__ void cleanup_device();
    __host__ void init_device(int width, int height);

    __host__ __device__ float unit();
    __host__ __device__ glm::vec2 square();
    __host__ __device__ glm::vec2 disk();
    __host__ __device__ glm::vec3 hemisphere();
    __host__ __device__ glm::vec3 hemisphere_cosine(float& pdf);
}
