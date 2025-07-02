
#pragma once

#include <cuda_runtime.h>
#include <algorithm>

#include "ray.h"

// min and max from math_functions.h in device code, but std::algorithm in host code
using namespace std;

class Bounds {
public:
    __host__ __device__ Bounds() {
        p_min = glm::vec3( INFINITY);
        p_max = glm::vec3(-INFINITY);
    }

    __host__ __device__ Bounds(const glm::vec3& a, const glm::vec3& b) :
        p_min(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)), 
        p_max(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)) {
    }

    __host__ __device__ glm::vec3 extent() const { return p_max - p_min; }

    __host__ __device__ glm::vec3 centroid() const { return 0.5f * (p_min + p_max); }

    // Returns the index of the largest dimension
    __host__ __device__ int max_extent_dim() const {
        const glm::vec3& dims = extent();
        if (dims.x > dims.y) {
            if (dims.x > dims.z) {
                return 0;
            }
        } else if (dims.y > dims.z) {
            return 1;
        }
        return 2;
    }

    __host__ __device__ float surface_area() const {
        const glm::vec3& dims = extent();
        return 2.f*(dims.x*dims.y + dims.x*dims.z + dims.y*dims.z);
    }

    __host__ __device__ float volume() const {
        const glm::vec3& dims = extent();
        return dims.x * dims.y * dims.z;
    }

    __host__ __device__ void enclose(const glm::vec3& p) {
        p_min = glm::vec3(min(p.x, p_min.x), min(p.y, p_min.y), min(p.z, p_min.z));
        p_max = glm::vec3(max(p.x, p_max.x), max(p.y, p_max.y), max(p.z, p_max.z));
    }

    __host__ __device__ void enclose(const Bounds& b) {
        enclose(b.p_min);
        enclose(b.p_max);
    }

    __host__ __device__ void transform(const glm::mat4& trans) {
        glm::vec3 a = glm::vec3(trans * glm::vec4(p_min, 1.f));
        glm::vec3 b = glm::vec3(trans * glm::vec4(p_max, 1.f));
        p_min = glm::vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
        p_max = glm::vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
    }

    // Any degenerate dimensions will not be normalized
    __host__ __device__ glm::vec3 normalize_point(glm::vec3 p) {
        glm::vec3 normalized = p - p_min;
        if (p_max.x > p_min.x)
            normalized.x /= p_max.x - p_min.x;
        if (p_max.y > p_min.y)
            normalized.y /= p_max.y - p_min.y;
        if (p_max.z > p_min.z)
            normalized.z /= p_max.z - p_min.z;
        return normalized;
    }

    __host__ __device__ glm::vec3 operator[](int i) const {
        return (i == 0) ? p_min : p_max;
    }
    __host__ __device__ glm::vec3& operator[](int i) {
        return (i == 0) ? p_min : p_max;
    }

    // Standard intersection, no precomputed values
    __host__ __device__ inline bool intersect(const Ray& r) const {
        float t_near = 0, t_far = r.max_t;
        for (int axis = 0; axis < 3; axis++) {
            const float inv_dir = 1.f / r.d[axis];

            float t0 = (p_min[axis] - r.o[axis]) * inv_dir;
            float t1 = (p_max[axis] - r.o[axis]) * inv_dir;

            if (t0 < t1) {
                if (t0 > t_near) t_near = t0;
                if (t1 < t_far ) t_far  = t1;
            } else {
                if (t1 > t_near) t_near = t1;
                if (t0 < t_far ) t_far  = t0;
            }

            if (t_far <= t_near)
                return false;
        }
        return true;
    }

    // Faster intersection with precomputed inverse direction and sign (from pbrt)
    __host__ __device__ inline bool intersect(const Ray& r, 
                                              const glm::vec3& inv_dir, 
                                              const int dir_sign[3]) const {
        const Bounds& b = *this;

        float t_min = (b[dir_sign[0]].x - r.o.x) * inv_dir.x;
        float t_max = (b[1 - dir_sign[0]].x - r.o.x) * inv_dir.x;
        float ty_min = (b[dir_sign[1]].y - r.o.y) * inv_dir.y;
        float ty_max = (b[1 - dir_sign[1]].y - r.o.y) * inv_dir.y;

        if (t_min > ty_max || ty_min > t_max) return false;
        if (ty_min > t_min) t_min = ty_min;
        if (ty_max < t_max) t_max = ty_max;

        float tz_min = (b[dir_sign[2]].z - r.o.z) * inv_dir.z;
        float tz_max = (b[1 - dir_sign[2]].z - r.o.z) * inv_dir.z;

        if (t_min > tz_max || tz_min > t_max) return false;
        if (tz_min > t_min) t_min = tz_min;
        if (tz_max < t_max) t_max = tz_max;

        return (t_min < r.max_t) && (t_max > 0);
    }

    glm::vec3 p_min;
    glm::vec3 p_max;
};

// Avoiding multiple definitions
#ifdef __CUDACC__
__host__ __device__ Bounds union_bounds(const Bounds& a, const Bounds& b) {
    glm::vec3 u_min(min(a.p_min.x, b.p_min.x), 
                    min(a.p_min.y, b.p_min.y), 
                    min(a.p_min.z, b.p_min.z));
    glm::vec3 u_max(max(a.p_max.x, b.p_max.x), 
                    max(a.p_max.y, b.p_max.y), 
                    max(a.p_max.z, b.p_max.z));
    return Bounds(u_min, u_max);
}
#endif
