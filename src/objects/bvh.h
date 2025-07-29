
#pragma once

#include <stdexcept>

#include <cuda_runtime.h>

#include "bounds.h"
#include "hit_record.h"
#include "sphere.h"

//#define DEBUG

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
    __host__ __device__ BVH(Primitive *_prims, int _nprims, BVHNode *_bvh, int _nnodes) 
        : prims(_prims), nprims(_nprims), bvh(_bvh), nnodes(_nnodes) {
    }

    __host__ __device__ Bounds bounds() const {
        return bvh[0].bbox;
    }

    // Hit should not be called on BVHs -- only included to satisfy the object interface
    __host__ __device__ bool hit(const Ray& r, HitRecord& rec) const { return false; }

    // Intersect is called on the top level BVH
    __host__ __device__ bool hit_if_if(const Ray& r, HitRecord& rec) const {
        bool hit = false;

        // Precomputing values for faster bbox intersection
        glm::vec3 inv_dir = 1.f / r.d;
        int dir_sign[3] = {int(inv_dir.x < 0), int(inv_dir.y < 0), int(inv_dir.z < 0)};

        int to_visit[64];      // Stack of nodes waiting to be checked
        BVH<Primitive> *roots[64];    // Stack of roots that ^ indexes.
        int to_visit_idx = 0;  // Current position in the stack
        int node_idx = 0;      // Index of the BVH node to check
        BVH<Primitive> *root = const_cast<BVH<Primitive> *>(this);

        while (true) {
            #ifdef DEBUG
            if (node_idx >= root->nnodes) {
                printf("Out of bounds node index.\n");
            }
            #endif
            const BVHNode *node = &(root->bvh[node_idx]);
            if (node->bbox.intersect(r, inv_dir, dir_sign)) {
                // Intersection with bounds, check node
                if (node->num_hitables > 0) {
                    // Leaf node, intersect with hitables
                    to_visit_idx--;
                    for (int i = 0; i < node->num_hitables; ++i) {
                        int prim_idx = node->start_idx + i;
                        #ifdef DEBUG
                        if (prim_idx >= root->nprims) {
                            printf("Out of bounds primitive index.\n");
                        }
                        #endif
                        const Primitive &prim = root->prims[prim_idx];
                        if (const BVH<Primitive> *tree = cuda::std::get_if<BVH<Primitive>>(&(prim.underlying))) {
                            roots[++to_visit_idx] = const_cast<BVH<Primitive> *>(tree);
                            to_visit[to_visit_idx] = 0;
                        } else {
                            hit |= prim.hit(r, rec);
                        }
                    }
                    if (to_visit_idx < 0)
                        break;  // Stack has been exhausted
                    node_idx = to_visit[to_visit_idx];
                    root = roots[to_visit_idx];
                } else {
                    // Interior node, traverse to near node and put far on the stack
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

    __host__ __device__ bool hit_while_while(const Ray& r, HitRecord& rec) const {
        bool hit = false;

        // Precomputing values for faster bbox intersection
        glm::vec3 inv_dir = 1.f / r.d;
        int dir_sign[3] = {int(inv_dir.x < 0), int(inv_dir.y < 0), int(inv_dir.z < 0)};

        int to_visit[64];      // Stack of nodes waiting to be checked
        BVH<Primitive> *roots[64];    // Stack of roots that ^ indexes.
        int to_visit_idx = 0;  // Current position in the stack
        int node_idx = 0;      // Index of the BVH node to check
        BVH<Primitive> *root = const_cast<BVH<Primitive> *>(this);

        while (true) {
            #ifdef DEBUG
            if (node_idx >= root->nnodes) {
                printf("Out of bounds node index.\n");
            }
            #endif
            // while current node is an interior node, traverse to next node
            const BVHNode *node = &(root->bvh[node_idx]);
            while (node->num_hitables == 0) {
                if (node->bbox.intersect(r, inv_dir, dir_sign)) {
                    if (dir_sign[node->axis]) {
                        roots[to_visit_idx] = root;
                        to_visit[to_visit_idx++] = node_idx + 1;
                        node_idx = node->one_idx;
                    } else {
                        roots[to_visit_idx] = root;
                        to_visit[to_visit_idx++] = node->one_idx;
                        node_idx = node_idx + 1;
                    }
                } else {
                    if (--to_visit_idx < 0)
                        return hit;
                    node_idx = to_visit[to_visit_idx];
                    root = roots[to_visit_idx];
                }
                node = &(root->bvh[node_idx]);
            }
            // while node has triangles to test, perform intersection test
            to_visit_idx--;
            int prim_offset = 0;
            do {
                int prim_idx = node->start_idx + prim_offset;
                #ifdef DEBUG
                if (prim_idx >= root->nprims) {
                    printf("Out of bounds primitive index.\n");
                }
                #endif
                const Primitive &prim = root->prims[prim_idx];
                if (const BVH<Primitive> *tree = cuda::std::get_if<BVH<Primitive>>(&(prim.underlying))) {
                    roots[++to_visit_idx] = const_cast<BVH<Primitive> *>(tree);
                    to_visit[to_visit_idx] = 0;
                } else {
                    hit |= prim.hit(r, rec);
                }
                ++prim_offset;
            } while (prim_offset < node->num_hitables);
            if (to_visit_idx < 0)
                return hit;  // Stack has been exhausted
            node_idx = to_visit[to_visit_idx];
            root = roots[to_visit_idx];
        }
        return hit;
    }

    __device__ bool hit_speculative(const Ray& r, HitRecord& rec) const {
        bool hit = false;

        // Precomputing values for faster bbox intersection
        glm::vec3 inv_dir = 1.f / r.d;
        int dir_sign[3] = {int(inv_dir.x < 0), int(inv_dir.y < 0), int(inv_dir.z < 0)};

        int to_visit[64];      // Stack of nodes waiting to be checked
        BVH<Primitive> *roots[64];    // Stack of roots that ^ indexes.
        int to_visit_idx = 0;  // Current position in the stack
        int node_idx = 0;      // Index of the BVH node to check
        BVH<Primitive> *root = const_cast<BVH<Primitive> *>(this);

        const BVHNode *leaf = nullptr;  // Buffered leaf node for speculative traversal

        while (true) {
            #ifdef DEBUG
            if (node_idx >= root->nnodes) {
                printf("Out of bounds node index.\n");
            }
            #endif
            // while current node is an interior node, traverse to next node
            const BVHNode *node = &(root->bvh[node_idx]);
            if (node->num_hitables == 0) {
                if (node->bbox.intersect(r, inv_dir, dir_sign)) {
                    if (dir_sign[node->axis]) {
                        roots[to_visit_idx] = root;
                        to_visit[to_visit_idx++] = node_idx + 1;
                        node_idx = node->one_idx;
                    } else {
                        roots[to_visit_idx] = root;
                        to_visit[to_visit_idx++] = node->one_idx;
                        node_idx = node_idx + 1;
                    }
                } else {
                    if (--to_visit_idx < 0)
                        return hit;
                    node_idx = to_visit[to_visit_idx];
                    root = roots[to_visit_idx];
                }
                node = &(root->bvh[node_idx]);
            }
            
            if (node->num_hitables != 0 && !leaf) {
                leaf = node;
                to_visit_idx--;
            }

            // wait until all threads have a leaf for intersection
            if (__any_sync(__activemask(), !leaf))
                continue;        

            // while node has triangles to test, perform intersection test
            int prim_offset = 0;
            do {
                int prim_idx = leaf->start_idx + prim_offset;
                #ifdef DEBUG
                if (prim_idx >= root->nprims) {
                    printf("Out of bounds primitive index.\n");
                }
                #endif
                const Primitive &prim = root->prims[prim_idx];
                if (const BVH<Primitive> *tree = cuda::std::get_if<BVH<Primitive>>(&(prim.underlying))) {
                    roots[++to_visit_idx] = const_cast<BVH<Primitive> *>(tree);
                    to_visit[to_visit_idx] = 0;
                } else {
                    hit |= prim.hit(r, rec);
                }
                ++prim_offset;
            } while (prim_offset < node->num_hitables);
            leaf = nullptr;

            if (to_visit_idx < 0)
                return hit;  // Stack has been exhausted
            node_idx = to_visit[to_visit_idx];
            root = roots[to_visit_idx];
        }
        return hit;
    }

    // Left public to allow deallocating externally
    Primitive *prims;
    BVHNode *bvh;
    int nprims;
    int nnodes;
};
