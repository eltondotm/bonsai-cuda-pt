#pragma once

#include <cuda/std/limits>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/matrix.hpp>

class Ray
{
    public:
        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const glm::vec3& _o, const glm::vec3& _d) 
            : o(_o), d(glm::normalize(_d)) {}

        __host__ __device__ glm::vec3 at(float t) const { 
            return o + t*d; 
        }
        
        __host__ __device__ void transform(const glm::mat4& trans) {
            o = glm::vec3(trans * glm::vec4(o, 1.f));
            d = glm::vec3(trans * glm::vec4(d, 0.f));
            float len = glm::length(d);
            max_t *= len;
            d /= len;
        }

        glm::vec3 o;
        glm::vec3 d;
        mutable float max_t = cuda::std::numeric_limits<float>::max();
};
