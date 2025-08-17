
#pragma once

#include <cuda_runtime.h>
#include <cuda/std/variant>

#include <glm/vec3.hpp>

#include "bounds.h"
#include "hit_record.h"
#include "sphere.h"
#include "triangle.h"
#include "square.h"
#include "object_list.h"
#include "bvh.h"

class Object {
public:
    __host__ __device__ Object(Sphere&& sphere, uint16_t m = 0, uint16_t t = 0)
        : underlying(cuda::std::move(sphere)), material(m), trans(t) {
    }
    __host__ __device__ Object(Triangle&& tri, uint16_t m = 0, uint16_t t = 0)
        : underlying(cuda::std::move(tri)), material(m), trans(t) {  
    }
    __host__ __device__ Object(Square&& square, uint16_t m = 0, uint16_t t = 0)
        : underlying(cuda::std::move(square)), material(m), trans(t) {
    }
    __host__ __device__ Object(BVH<Object>&& bvh, uint16_t m = 0, uint16_t t = 0)
        : underlying(cuda::std::move(bvh)), material(m), trans(t) {
    }

    __host__ __device__ Bounds bounds() const {
        return cuda::std::visit(overloaded{[](const auto& o) { return o.bounds(); }}, underlying);
    }
    __host__ __device__ Bounds bounds(const Transform& t) const {
        Bounds b = cuda::std::visit(overloaded{[](const auto& o) { return o.bounds(); }}, underlying);
        b.transform(t.trans);
        return b;
    }

    __host__ __device__ bool hit(const Ray& r, HitRecord& rec, const Transform& t) const {
        Ray r_local(r);
        r_local.transform(t.itrans);
        bool hit = cuda::std::visit(overloaded{
            [&r_local, &rec](const auto& o) { return o.hit(r_local, rec); }
        }, underlying);
        if (hit) {
            rec.material = material;
            rec.transform(t.trans, glm::transpose(t.itrans), r.o);
            r.max_t = rec.time;
        }
        return hit;
    }
    
    cuda::std::variant<Triangle, Sphere, Square, BVH<Object>> underlying;

    uint16_t material;
    uint16_t trans;
};
