
#pragma once

#include <cuda_runtime.h>
#include <cuda/std/variant>

#include <glm/common.hpp>

#include "util/random.h"

struct BSDFSample {
    glm::vec3 emission;
    glm::vec3 attenuation;
    glm::vec3 direction;
    float pdf;
};

struct Lambertian {
    glm::vec3 albedo;
    BSDFSample sample(glm::vec3 wo) const;
};

struct Metallic {
    glm::vec3 albedo;
    glm::vec3 fuzz;
    BSDFSample sample(glm::vec3 wo) const;
};

struct Glass {
    glm::vec3 ior;
    BSDFSample sample(glm::vec3 wo) const;
};

struct Emissive {
    glm::vec3 emission;
    BSDFSample sample(glm::vec3 wo) const;
};

using Material = cuda::std::variant<Lambertian, Metallic, Glass, Emissive>;

// Lambertian diffuse
glm::vec3 sample_diffuse(glm::vec3 wo) {
    float pdf = 0.f;
    glm::vec3 wi = rng::hemisphere_cosine(pdf);

}

BSDFSample Lambertian::sample(glm::vec3 wo) const {
    float pdf = 0.f;
    glm::vec3 wi = rng::hemisphere_cosine(pdf);

    BSDFSample sample;
    sample.emission = glm::vec3(0.f);
    sample.attenuation = albedo / glm::radians(180.f);
    sample.direction = wi;
    sample.pdf = pdf;
    return sample;
}

BSDFSample Metallic::sample(glm::vec3 wo) const {
    return BSDFSample();
}

BSDFSample Glass::sample(glm::vec3 wo) const {
    return BSDFSample();
}

BSDFSample Emissive::sample(glm::vec3 wo) const {
    BSDFSample sample;
    sample.emission = emission;
    sample.attenuation = glm::vec3(0.f);
    sample.direction = glm::vec3(0.f, 1.f, 0.f);
    return sample;
}

template<class... Ts> struct overloaded : Ts ... { using Ts::operator() ...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

BSDFSample sample(const Material& m, glm::vec3 wo) {
    return cuda::std::visit(overloaded{
        [&wo](const auto& mat) { return mat.sample(wo); }
    }, m);
}
