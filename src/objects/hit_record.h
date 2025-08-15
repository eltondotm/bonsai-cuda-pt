
#pragma once

#include <glm/vec3.hpp>

#include "materials.h"

struct HitRecord {
    glm::vec3    position;
    glm::vec3    normal;
    float        time;
    unsigned int material;

    // Origin should be pre-transform ray origin
    __host__ __device__ void transform(const glm::mat4& transform, const glm::mat4& norm, const glm::vec3& origin) {
        position = transform * glm::vec4(position, 1.f);
        normal = glm::normalize(norm * glm::vec4(normal, 0.f));
        time = glm::length(position - origin);
    }
};
