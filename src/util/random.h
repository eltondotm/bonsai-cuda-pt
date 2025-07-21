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

#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/trigonometric.hpp>

namespace rng {

///////////////////////////////////////
// Host generator management
///////////////////////////////////////

// Stores current position in the buffer
static size_t i;

// Refills buffer and resets current index
__host__ void generate_buffer();

// Destroys generator and deallocates buffer
__host__ void cleanup_host();

// Creates generator and allocates buffers, reinitializes on future calls
__host__ void init_host(size_t buffer_size=1024);


///////////////////////////////////////
// Device generator management
///////////////////////////////////////

__host__ void cleanup_device();

// Reinitialize if dimensions change
__host__ void init_device(int width, int height);


///////////////////////////////////////
// Common random functions
///////////////////////////////////////

// Uniform float in (0, 1]
__host__ __device__ float unit();

// Uniformly samples a unit square (0, 1]^2
__host__ __device__ glm::vec2 square();

// Uniformly samples a unit triangle
__host__ __device__ glm::vec2 triangle();

// Uniformly samples a unit disk (Shirley's method for reduced distortion)
__host__ __device__ glm::vec2 disk();

// Uniformly samples a unit hemisphere
__host__ __device__ glm::vec3 hemisphere(float& pdf);

// Samples hemisphere with cosine weighting
__host__ __device__ glm::vec3 hemisphere_cosine(float& pdf);

}  // namespace rng
