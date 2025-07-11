
#pragma once

#include <cuda_runtime.h>

#include "bounds.h"
#include "hit_record.h"

template<typename Primitive> class List {
public:
    __host__ __device__ List(Primitive *_prims, int _nprims) 
        : prims(_prims), nprims(_nprims) {
        Bounds b;
        for (int i = 0; i < nprims; ++i) {
            b.enclose(prims[i].bounds());
        }
        bbox = b;
    }

    __host__ __device__ Bounds bounds() const {
        return bbox;
    }

    // Same as BVH, hit should never be called on a list to avoid recursion
    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const {
        return false;
    }

    __host__ __device__ bool intersect(const Ray& r, HitRecord& rec) const {
        bool hit = false;
        for (int i = 0; i < nprims; ++i) {
            // Assumes ray max_t is updated by all hit methods (other than bounds)
            if (prims[i].hit(r, rec)) {
                hit = true;
            }
        }
        return hit;
    }

private:
    Bounds bbox;
    Primitive *prims;
    int nprims;
};
