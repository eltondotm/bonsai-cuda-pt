
#pragma once

#include <glm/vec3.hpp>

#include "materials.h"

struct HitRecord {
    glm::vec3 position;
    glm::vec3 normal;
    float     time;
    Material *material;
};
