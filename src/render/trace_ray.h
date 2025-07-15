
#pragma once

#include <cuda_runtime.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "util/random_declarations.h"
#include "scene.h"

#define MAX_BOUNCES 10
#define LIGHT_SAMPLES 2
#define EPS_F 0.0001f

// Constructs a transformation matrix that rotates the input direction to (0, 1, 0)
inline __host__ __device__ glm::mat4 rotate_to(glm::vec3 dir) {
    // glm::orientation does not handle special cases of same or opposite direction
    if(abs(dir.y - 1.0f) < EPS_F)
        return glm::mat4(1.f);
    else if(abs(dir.y + 1.0f) < EPS_F)
        return glm::mat4{glm::vec4{1.0f,  0.0f,  0.0f,  0.0f}, 
                         glm::vec4{0.0f, -1.0f,  0.0f,  0.0f},
                         glm::vec4{0.0f,  0.0f,  1.0f,  0.0}, 
                         glm::vec4{0.0f,  0.0f,  0.0f,  1.0f}};
    return glm::orientation(dir, glm::vec3{0.f, 1.f, 0.f});
}

__host__ __device__ glm::vec3 trace_ray(const Ray& r, const Scene& scene) {
    Ray ray(r);
    const BVH<Object> *geo = scene.geometry;
    glm::vec3 radiance(0.0f);
    glm::vec3 throughput(1.f);
    HitRecord rec;

    for (int i = 0; i < MAX_BOUNCES; ++i) {
        if (geo->intersect(ray, rec)) {
            //return rec.normal*0.5f+0.5f;  // Normal visualization

            // Transforming normal vector to (0, 1, 0) for easier material sampling
            glm::mat4 normal_to_world = rotate_to(rec.normal);
            glm::mat4 world_to_normal = glm::transpose(normal_to_world);
            glm::vec3 wo = world_to_normal * glm::vec4(glm::normalize(ray.o - rec.position), 0.f);

            // Sampling hit material
            BSDFSample sample = sample_bsdf(*rec.material, wo);

            if (i == 0 && sample.emission != glm::vec3(0.f))
                return sample.emission;
            radiance += sample.emission * throughput;

            // Create new ray
            float cos_theta_i = sample.direction.y;
            glm::vec3 wi = normal_to_world * glm::vec4(sample.direction, 0.f);
            ray = Ray(rec.position, wi);
            ray.o = ray.at(EPS_F);

            throughput *= sample.attenuation * cos_theta_i / sample.pdf;
            if (throughput == glm::vec3(0.f)) break;
        } else {
            break;
        }
    }  
    return radiance;
}

/* Monte Carlo with light sampling (not working)
__host__ __device__ glm::vec3 trace_ray(const Ray& r, const Scene& scene) {
    Ray ray(r);
    const BVH<Object> *geo = scene.geometry;
    glm::vec3 radiance(0.0f);
    glm::vec3 throughput(1.f);
    HitRecord rec;

    for (int i = 0; i < MAX_BOUNCES; ++i) {
        if (geo->intersect(ray, rec)) {
            //return rec.normal*0.5f+0.5f;  // Normal visualization

            // Transforming normal vector to (0, 1, 0) for easier material sampling
            glm::mat4 normal_to_world = rotate_to(rec.normal);
            glm::mat4 world_to_normal = glm::transpose(normal_to_world);
            glm::vec3 wo = world_to_normal * glm::vec4(glm::normalize(ray.o - rec.position), 0.f);

            // Sampling lights
            glm::vec3 light_radiance(0.f);
            for (int j = 0; j < scene.n_emitters; ++j) {
                const Triangle& light = cuda::std::get<Triangle>(scene.emitters[j]->underlying);
                for (int _ = 0; _ < LIGHT_SAMPLES; ++_) {
                    glm::vec3 sample_pos = light.point_at_barycentric(rng::triangle());
                    glm::vec3 sample_vec = sample_pos - rec.position;
                    glm::vec3 sample_dir = world_to_normal * glm::vec4(glm::normalize(sample_vec), 0.f);
                    float     sample_dst_sqr = glm::dot(sample_vec, sample_vec);
                    float     sample_dst = sqrt(sample_dst_sqr);
                    float     sample_pdf = sample_dst_sqr / light.area();

                    glm::vec3 in_dir = world_to_normal * glm::vec4(sample_dir, 0.f);

                    // If the light is below the horizon, ignore it
                    float cos_theta = in_dir.y;
                    if(cos_theta <= 0.0f) continue;
                    sample_pdf /= cos_theta;

                    glm::vec3 light_atten = glm::vec3(1.f) / glm::radians(180.f);

                    Ray shadow_ray = Ray(rec.position, sample_dir);
                    shadow_ray.o = shadow_ray.at(EPS_F);
                    shadow_ray.max_t = sample_dst - EPS_F;

                    HitRecord temp_rec;
                    if (!geo->intersect(shadow_ray, temp_rec))
                        light_radiance += (cos_theta / (LIGHT_SAMPLES * sample_pdf)) * glm::vec3(100.f) * light_atten;
                }  
            }
            radiance += light_radiance * throughput;

            // Sampling hit material
            BSDFSample sample = sample_bsdf(*rec.material, wo);

            if (i == 0 && sample.emission != glm::vec3(0.f))
                return sample.emission;
            //attenuation *= (1.0f + sample.emission);

            // Create new ray
            float cos_theta_i = sample.direction.y;
            glm::vec3 wi = normal_to_world * glm::vec4(sample.direction, 0.f);
            ray = Ray(rec.position, wi);
            ray.o = ray.at(EPS_F);

            throughput *= sample.attenuation * cos_theta_i / sample.pdf;

            // Potentially terminate
            float q = 1.0f - throughput.x;
            if (rng::unit() < q) break;

            throughput /= (1.0f - q);
        } else {
            break;
        }
    }  
    return radiance;
}*/
