
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
    __host__ __device__ Object(Sphere&& sphere, unsigned int m = 0, const glm::mat4& t = glm::mat4(1.f))
        : underlying(cuda::std::move(sphere)), material(m), trans(t), itrans(glm::inverse(t)) {
    }
    __host__ __device__ Object(Triangle&& tri, unsigned int m = 0, const glm::mat4& t = glm::mat4(1.f))
        : underlying(cuda::std::move(tri)), material(m), trans(t), itrans(glm::inverse(t)) {  
    }
    __host__ __device__ Object(Square&& square, unsigned int m = 0, const glm::mat4& t = glm::mat4(1.f))
        : underlying(cuda::std::move(square)), material(m), trans(t), itrans(glm::inverse(t)) {
    }
    __host__ __device__ Object(BVH<Object>&& bvh, unsigned int m = 0, const glm::mat4& t = glm::mat4(1.f))
        : underlying(cuda::std::move(bvh)), material(m), trans(t), itrans(glm::inverse(t)) {
    }

    __host__ __device__ Bounds bounds() const {
        Bounds b = cuda::std::visit(overloaded{[](const auto& o) { return o.bounds(); }}, underlying);
        b.transform(trans);
        return b;
    }

    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const {
        Ray r_local(r);
        r_local.transform(itrans);
        bool hit = cuda::std::visit(overloaded{
            [&r_local, &rec](const auto& o) { return o.hit(r_local, rec); }
        }, underlying);
        if (hit) {
            rec.material = material;
            rec.transform(trans, glm::transpose(itrans), r.o);
            r.max_t = rec.time;
        }
        return hit;
    }
    
    cuda::std::variant<Triangle, Sphere, Square, BVH<Object>> underlying;

    unsigned int material;
    glm::mat4 trans;
    glm::mat4 itrans;
};
