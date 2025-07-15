
#pragma once

#include <cuda_runtime.h>
#include <cuda/std/variant>

#include <miniScene/Scene.h>

#include <glm/vec3.hpp>
#include <glm/trigonometric.hpp>

#include "util/random_declarations.h"

template<class... Ts> struct overloaded : Ts ... { using Ts::operator() ...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct BSDFSample {
    glm::vec3 emission;
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

__host__ __device__ BSDFSample sample_bsdf(const Material *m, glm::vec3 wo);

#ifdef __CUDACC__
__host__ __device__ BSDFSample Lambertian::sample(glm::vec3 wo) const {
    float pdf = 0.f;
    glm::vec3 wi = rng::hemisphere_cosine(pdf);

    BSDFSample sample;
    sample.emission = glm::vec3(0.f);
    sample.attenuation = albedo / glm::radians(180.f);
    sample.direction = wi;
    sample.pdf = pdf;
    return sample;
}

__host__ __device__ BSDFSample Metallic::sample(glm::vec3 wo) const {
    return BSDFSample();
}

__host__ __device__ BSDFSample Glass::sample(glm::vec3 wo) const {
    return BSDFSample();
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
#endif
