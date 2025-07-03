
#pragma once

#include <cuda_runtime.h>
#include <glm/common.hpp>
#include <glm/vec3.hpp>

#include "ray.h"
#include "object.h"

__host__ __device__ glm::vec3 trace_ray(const Ray& r, BVH<Object> *obj) {
    glm::vec3 light_dir = glm::normalize(glm::vec3(-0.4f, -0.6f, -1.f));

    HitRecord rec;
    if (obj->hit(r, rec)) {
        float intensity = glm::dot(rec.normal, -light_dir);
        return glm::vec3(glm::clamp(intensity, 0.f, 1.f));
    }

    float a = 0.5f*r.d.y+0.5f;
    return (1.f-a)*glm::vec3(1.f) + a*glm::vec3(0.5f, 0.7f, 1.f);
}
