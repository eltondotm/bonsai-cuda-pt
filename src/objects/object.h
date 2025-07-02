
#pragma once

#include <cuda_runtime.h>
#include <cuda/std/variant>

#include <glm/vec3.hpp>

#include "bounds.h"
#include "hit_record.h"
#include "sphere.h"
#include "object_list.h"
#include "bvh.h"

template<class... Ts> struct overloaded : Ts ... { using Ts::operator() ...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

class Object {
public:
    __host__ __device__ Object(Sphere&& sphere)
        : underlying(cuda::std::move(sphere)) {
    }
    __host__ __device__ Object(List<Object>&& list)
        : underlying(cuda::std::move(list)) {
    }
    __host__ __device__ Object(BVH<Object>&& bvh)
        : underlying(cuda::std::move(bvh)) {
    }

    __host__ __device__ Bounds bounds() const {
        return cuda::std::visit(overloaded{
            [](const auto& o) { return o.bounds(); }
        }, underlying);
    }

    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const {
        return cuda::std::visit(overloaded{
            [&r, &rec](const auto& o) { return o.hit(r, rec); }
        }, underlying);
    }
    
private:
    cuda::std::variant<Sphere, List<Object>, BVH<Object>> underlying;
};
