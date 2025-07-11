
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

// Not Disney BSDF, this is the material miniScene calls "DisneyMaterial"
struct miniDisney {
    glm::vec3 emission;
    glm::vec3 albedo;
    float metallic;
    float roughness;
    float transmission;
    float ior;
};

using Material = cuda::std::variant<miniDisney>;

// Lambertian diffuse
glm::vec3 sample_diffuse(glm::vec3 wo) {
    float pdf = 0.f;
    glm::vec3 wi = rng::hemisphere_cosine(pdf);

}

// BSDF sampling functiosn to pass to std::visit
struct Sample{
    BSDFSample operator() (const miniDisney& m) { 
        float pdf = 0.f;
        glm::vec3 wi = rng::hemisphere_cosine(pdf);

        BSDFSample sample;
        sample.emission = m.emission;
        sample.attenuation = m.albedo / glm::radians(180.f);
        sample.pdf = pdf;
        return sample;
    }
};
