
#pragma once

#include <cuda_runtime.h>

#include "bounds.h"
#include "hit_record.h"
#include "sphere.h"

// Binary tree node for use in traversal
struct alignas(32) BVHNode {
    Bounds bbox;
    union {
        int one_idx;  // Second child for interior nodes
        int start_idx;  // Leaf prim start index
    };
    uint16_t num_hitables;  // Nonzero for leaf nodes;
    uint8_t axis;  // Which axis was this node split along?
};

template<typename Primitive> class BVH {
public:
    __host__ __device__ BVH(Primitive *_prims, int _nprims, BVHNode *_bvh) 
        : prims(_prims), nprims(_nprims), bvh(_bvh) {
        bbox = bvh[0].bbox;
    }

    __host__ __device__ Bounds bounds() const {
        return bbox;
    }

    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const {
        bool hit = false;

        // Precomputing values for faster bbox intersection
        glm::vec3 inv_dir = 1.f / r.d;
        int dir_sign[3] = {int(inv_dir.x < 0), int(inv_dir.y < 0), int(inv_dir.z < 0)};

        int to_visit[64];      // Stack of nodes waiting to be checked
        BVHNode *roots[64];    // Stack of roots that ^ indexes.
        int to_visit_idx = 0;  // Current position in the stack
        int node_idx = 0;      // Index of the BVH node to check
        BVHNode *root = this->bvh; // BVHNode list to check.

        while (true) {
            const BVHNode *node = &root[node_idx];
            if (node->bbox.intersect(r, inv_dir, dir_sign)) {
                // Intersection with bounds, check node
                if (node->num_hitables > 0) {
                    // Leaf node, intersect with hitables
                    to_visit_idx--;
                    for (int i = 0; i < node->num_hitables; ++i) {
                        const Primitive &prim = prims[node->start_idx + i];
                        if (const BVH<Primitive> *tree = cuda::std::get_if<BVH<Primitive>>(&(prim.underlying))) {
                            roots[++to_visit_idx] = tree->bvh;
                            to_visit[to_visit_idx] = 0;
                        } else if (const Sphere *sphere = cuda::std::get_if<Sphere>(&(prim.underlying))) {
                            hit |= sphere->hit(r, rec);
                        } else {
                            // throw error
                        }
                    }
                    // No intersection, move on to next node
                    if (to_visit_idx <= 0)
                        break;  // Stack has been exhausted
                    node_idx = to_visit[to_visit_idx];
                    root = roots[to_visit_idx];
                } else {
                    // Interior node, traverse to near node and put far on the stack
                    //printf("Second child idx: %d\n", node->one_idx);
                    if (dir_sign[node->axis]) {
                        roots[to_visit_idx] = root;
                        to_visit[to_visit_idx++] = node_idx + 1;
                        node_idx = node->one_idx;
                    } else {
                        roots[to_visit_idx] = root;
                        to_visit[to_visit_idx++] = node->one_idx;
                        node_idx = node_idx + 1;
                    }
                }
            } else {
                if (to_visit_idx == 0)
                    break;
                node_idx = to_visit[--to_visit_idx];
                root = roots[to_visit_idx];
            }
        }
        return hit; 
    }

private:
    Bounds bbox;
    Primitive *prims;
    int nprims;
    BVHNode *bvh;
};
