
#pragma once

#include <cuda_runtime.h>

#include <glm/vec3.hpp>
#include <miniScene/Scene.h>

#include "bounds.h"
#include "hit_record.h"

struct TriangleMesh {
    glm::vec3  *vertices;
    glm::vec3  *normals;

    bool has_normals = false;
};

class Triangle {
public:
    __host__ __device__ Triangle(TriangleMesh *tri_mesh, glm::ivec3 vert_idxs) 
        : mesh(tri_mesh), idx(vert_idxs) {
    }

    __host__ __device__ Bounds bounds() const {
        Bounds b;
        b.enclose(mesh->vertices[idx.x]);
        b.enclose(mesh->vertices[idx.y]);
        b.enclose(mesh->vertices[idx.z]);
        return b;
    }

    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const {
        const glm::vec3& p0 = mesh->vertices[idx.x];
        const glm::vec3& p1 = mesh->vertices[idx.y];
        const glm::vec3& p2 = mesh->vertices[idx.z];

        // Moller Trumbore algorithm
        glm::vec3 p = r.o - p0;
        glm::vec3 ab = p1 - p0;
        glm::vec3 ac = p2 - p0;

        // Compute cross products for determinant + Cramer's rule
        glm::vec3 p_cross_ab = glm::cross(p, ab);
        glm::vec3 dir_cross_ac = glm::cross(r.d, ac);
        float det = dot(ab, dir_cross_ac);
        float inv_det = 1.0f / det;

        // Find u and v for barycentric coordinates, return no hit if not within triangle
        float u = glm::dot(p, dir_cross_ac) * inv_det;
        if(u < 0 || u > 1) return false;
        float v = glm::dot(r.d, p_cross_ab) * inv_det;
        if(v < 0 || u + v > 1) return false;

        // If we make it past the u and v checks, there is a hit
        float t = dot(ac, p_cross_ab) * inv_det;
        if(t < 0|| t > r.max_t) return false;
        r.max_t = t;

        glm::vec3 norm;
        if (mesh->has_normals) {
            // Calculate normal from barycentric weights
            glm::vec3 norm_a = mesh->normals[idx.r] * (1 - u - v);
            glm::vec3 norm_b = mesh->normals[idx.s] * u;
            glm::vec3 norm_c = mesh->normals[idx.t] * v;
            norm = norm_a + norm_b + norm_c;
        } else {
            // Default to plane normal (not smooth)
            norm = glm::cross(ab, ac);
        }
        

        // Flip uv for back triangle
        // if(ab.x > 0 || ab.z > 0) {
        //     u = 1.0f - u;
        //     v = 1.0f - v;
        // }
        
        rec.position = r.at(t);
        rec.normal = norm;
        rec.time = t;
        return true;
    }

private:
    TriangleMesh *mesh;
    glm::ivec3 idx;
};
