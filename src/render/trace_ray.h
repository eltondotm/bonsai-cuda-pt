
#pragma once

#include <cuda_runtime.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "ray.h"
#include "object.h"

__host__ __device__ glm::vec3 trace_ray(const Ray& r, BVH<Object> *obj) {
    glm::vec3 light_dir = glm::normalize(glm::vec3(-0.4f, -0.6f, -1.f));

    HitRecord rec;
    if (obj->intersect(r, rec)) {
        return rec.normal*0.5f+0.5f;
    }

    return glm::vec3(0.f);
}
