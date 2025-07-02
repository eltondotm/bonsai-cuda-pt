
#ifndef SPHEREH
#define SPHEREH

#include <cuda_runtime.h>
#include <cuda/std/utility>

#include "bounds.h"
#include "hit_record.h"

class Sphere {
public:
    __host__ __device__ Sphere() 
        : radius(1.f), center(0.f) { 
        bbox = Bounds(glm::vec3(-radius) + center, glm::vec3(radius) + center); 
    }
    __host__ __device__ Sphere(float _radius, const glm::vec3& _center) 
        : radius(_radius), center(_center) {
        bbox = Bounds(glm::vec3(-radius) + center, glm::vec3(radius) + center);
    }

    __host__ __device__ Bounds bounds() const { 
        return bbox;
    }
    
    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const {
        glm::vec3 oc = r.o - center;
        float a = 1.f;
        float b = 2.f * glm::dot(oc, r.d);
        float c = glm::dot(oc, oc) - (radius * radius);

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
        rec.normal = glm::normalize(rec.position - center);
        rec.time = t0;
        return true;
    }

private:
    Bounds bbox;
    float radius;
    glm::vec3 center;
};

#endif
