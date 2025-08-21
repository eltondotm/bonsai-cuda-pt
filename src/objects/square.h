
#pragma once

#include <cuda_runtime.h>
#include <cuda/std/utility>

#include "bounds.h"
#include "hit_record.h"

class Square {
public:
    __host__ __device__ Square() {}

    __host__ __device__ Bounds bounds() const { 
        return Bounds(glm::vec3(-1.f, -1.f, -0.0001f), glm::vec3(1.f, 1.f, 0.0001f));
    }
    
    // Ray is transformed so the intersection is with a unit square
    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const {
        // Time of intersection with the plane of the rectangle
        float t = -r.o.z / r.d.z;
        if (t < 0.f || t > r.max_t) return false;

        // Barycentrics determined by x and y coordinates of hit pos
        glm::vec3 pos = r.at(t);
        float u = pos.x*0.5f+0.5f;
        float v = pos.y*0.5f+0.5f;
        if (u < 0.f || u > 1.f || v < 0.f || v > 1.f) return false;

        rec.position = pos;
        rec.normal = r.d.z > 0 ? glm::vec3(0.f, 0.f, -1.f) : glm::vec3(0.f, 0.f, 1.f);
        rec.time = t;
        return true;
    }
};
