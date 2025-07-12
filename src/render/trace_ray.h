
#pragma once

#include <cuda_runtime.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "util/random_declarations.h"
#include "ray.h"
#include "object.h"

#define MAX_BOUNCES 10

inline __host__ __device__ glm::mat4 rotate_to(glm::vec3 dir) {
    if(std::abs(dir.y - 1.0f) < 0.001f)
        return glm::mat4(1.f);
    else if(std::abs(dir.y + 1.0f) < 0.001f)
        return glm::mat4{glm::vec4{1.0f, 0.0f, 0.0f, 0.0f}, glm::vec4{0.0f, -1.0f, 0.0f, 0.0f},
                    glm::vec4{0.0f, 0.0f, 1.0f, 0.0}, glm::vec4{0.0f, 0.0f, 0.0f, 1.0f}};
    glm::vec3 x = glm::normalize(glm::cross(dir, glm::vec3{0.0f, 1.0f, 0.0f}));
    glm::vec3 z = glm::normalize(glm::cross(x, dir));
    return glm::mat4{glm::vec4{x, 0.0f}, glm::vec4{dir, 0.0f}, glm::vec4{z, 0.0f}, glm::vec4{0.0f, 0.0f, 0.0f, 1.0f}};
}

__host__ __device__ glm::vec3 trace_ray(const Ray& r, BVH<Object> *obj) {
    Ray ray(r);
    glm::vec3 attenuation(1.f);
    HitRecord rec;

    for (int i = 0; i < MAX_BOUNCES; ++i) {
        if (obj->intersect(ray, rec)) {
            //return rec.normal*0.5f+0.5f;  // Normal visualization

            glm::mat4 normal_to_world = rotate_to(rec.normal);//glm::orientation(rec.normal, glm::vec3(0.f, 1.f, 0.f));
            glm::mat4 world_to_normal = glm::transpose(world_to_normal);

            //return normal_to_world*glm::vec4(rec.normal, 0.f)*0.5f+0.5f;

            glm::vec3 wi = rng::hemisphere();
            wi = normal_to_world * glm::vec4(wi, 0.f);
            ray = Ray(rec.position, wi);
            ray.o = ray.at(0.001f);

            attenuation *= 0.7f;
        } else {
            attenuation *= glm::vec3(0.8f, 0.8f, 0.8f);
            break;
        }
    }  

    return attenuation;
}
