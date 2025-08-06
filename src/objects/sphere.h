
#pragma once

#include <cuda_runtime.h>
#include <cuda/std/utility>

#include "bounds.h"
#include "hit_record.h"

class Sphere {
public:
    __host__ __device__ Sphere() 
        : radius(1.f) { 
    }
    __host__ __device__ Sphere(float _radius) 
        : radius(_radius) {
    }

    __host__ __device__ Bounds bounds() const { 
        return Bounds(glm::vec3(-radius), glm::vec3(radius));
    }
    
    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const {
        float a = 1.f;
        float b = 2.f * glm::dot(r.o, r.d);
        float c = glm::dot(r.o, r.o) - (radius * radius);

        float discriminant = (b * b) - (4.f * a * c);
        if (discriminant < 0) return false;
        float sqrt_discr = sqrt(discriminant);
        
        float t0, t1;
        float q;
        if (b > 0) q = -0.5f * (b + sqrt_discr);
        else q = -0.5f * (b - sqrt_discr);
        t0 = q / a;
        t1 = c / q;
        if (t0 > t1) cuda::std::swap(t0, t1);

        if (t0 < 0 || t0 > r.max_t) {
            t0 = t1;
            if (t0 < 0 || t0 > r.max_t) return false;
        }

        r.max_t = t0;

        rec.position = r.at(t0);
        rec.normal = glm::normalize(rec.position);
        rec.time = t0;
        return true;
    }

private:
    float radius;
};
