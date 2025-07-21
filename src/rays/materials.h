
#pragma once

#include <cuda_runtime.h>
#include <cuda/std/variant>
#include <cuda/std/utility>

#include <glm/vec3.hpp>
#include <glm/trigonometric.hpp>

#include "util/random.h"

template<class... Ts> struct overloaded : Ts ... { using Ts::operator() ...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct BSDFSample {
    glm::vec3 emission = glm::vec3(0.f);
    glm::vec3 attenuation;
    glm::vec3 direction;
    float pdf;
};

struct Lambertian {
    glm::vec3 albedo;
    __host__ __device__ BSDFSample sample(glm::vec3 wo) const;
};

struct Metallic {
    glm::vec3 albedo;
    float fuzz;
    __host__ __device__ BSDFSample sample(glm::vec3 wo) const;
};

struct Glass {
    float ior;
    __host__ __device__ BSDFSample sample(glm::vec3 wo) const;
};

struct Emissive {
    glm::vec3 emission;
    __host__ __device__ BSDFSample sample(glm::vec3 wo) const;
};

using Material = cuda::std::variant<Lambertian, Metallic, Glass, Emissive>;

__host__ __device__ BSDFSample sample_bsdf(const Material& m, glm::vec3 wo);

#ifdef __CUDACC__
// Faster reflect than glm since normal vector is always (0, 1, 0)
inline __host__ __device__ glm::vec3 reflect(glm::vec3 dir) {
    return glm::vec3(dir.x, -dir.y, dir.z);
}

// Schlick approximation of Fresnel reflection
inline __host__ __device__ float schlick(float cos, float ior) {
    float r0 = (1.f - ior) / (1.f + ior);
    r0 = r0*r0;
    float cos_term = (1.0f - cos);
    cos_term = (cos_term * cos_term) * (cos_term * cos_term) * cos_term;
    return r0 + (1.f-r0) * cos_term;
}

__host__ __device__ BSDFSample Lambertian::sample(glm::vec3 wo) const {
    float pdf = 0.f;
    glm::vec3 wi = rng::hemisphere_cosine(pdf);

    BSDFSample sample;
    sample.attenuation = albedo / glm::radians(180.f);
    sample.direction = wi;
    sample.pdf = pdf;
    return sample;
}

__host__ __device__ BSDFSample Metallic::sample(glm::vec3 wo) const {
    glm::vec3 wi = reflect(wo);

    BSDFSample sample;
    sample.attenuation = glm::vec3(1.0f);
    sample.direction = wi;
    sample.pdf = 1.f;
    return sample;
}

__host__ __device__ BSDFSample Glass::sample(glm::vec3 wo) const {
    float eta, cos;
    glm::vec3 normal;
    if (wo.y > 0) {
        normal = glm::vec3(0.f, -1.f, 0.f);
        eta = ior;
        cos = sqrt(1.f - ior * ior * (1.f - wo.y * wo.y));
    } else {
        normal = glm::vec3(0.f, 1.f, 0.f);
        eta = 1.f / ior;
        cos = -wo.y;
    }

    glm::vec3 wi = glm::refract(wo, normal, eta);
    if (wi == glm::vec3(0.f)) {
        wi = reflect(wo);
    } else {
        float fr = schlick(cos, ior);
        if (rng::unit() < fr) {
            wi = reflect(wo);
        }
    }

    // float eta = wo.y < 0 ? 1.0f / ior : ior;
    // glm::vec3 wi = glm::refract(wo, glm::vec3(0.f, 1.f, 0.f), eta);

    // // refract returns 0-vector on total internal reflection
    // if (wi == glm::vec3(0.f)) {
    //     wi = reflect(wo);
    // } else {
    //     float cos = abs(glm::dot(wo, wi));
    //     float fr = schlick(cos, eta);
    //     if (rng::unit() < fr)
    //         wi = reflect(wo);
    // }

    BSDFSample sample;
    sample.attenuation = glm::vec3(1.f);
    sample.direction = wi;
    sample.pdf = 1.f;
    return sample;
}

__host__ __device__ BSDFSample Emissive::sample(glm::vec3 wo) const {
    BSDFSample sample;
    sample.emission = emission;
    sample.attenuation = glm::vec3(0.f);
    sample.direction = glm::vec3(0.f, 1.f, 0.f);
    return sample;
}

__host__ __device__ BSDFSample sample_bsdf(const Material& m, glm::vec3 wo) {
    return cuda::std::visit(overloaded{
        [&wo](const auto& mat) { return mat.sample(wo); }
    }, m);
}

__host__ __device__ bool is_discrete(const Material& m) {
    return cuda::std::visit(overloaded{
        [](const Metallic& mat) { return true;  },
        [](const Glass&    mat) { return true;  },
        [](const auto&     mat) { return false; }
    }, m);
}
#endif
