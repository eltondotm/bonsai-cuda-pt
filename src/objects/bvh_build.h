
#pragma once

#include <vector>

#include "object.h"

Bounds union_bounds(const Bounds& a, const Bounds& b);

// Binary tree node for use in construction
struct BVHTreeNode {
    void create_leaf(const Bounds& b, int start, int num) {
        bbox = b;
        zero = nullptr;
        one  = nullptr;
        start_idx = start;
        num_hitables = num;
    }

    void create_interior(BVHTreeNode *_zero, BVHTreeNode *_one, int axis) {
        bbox = union_bounds(_zero->bbox, _one->bbox);
        zero = _zero;
        one  = _one;
        split_axis = axis;
    }

    Bounds bbox;
    BVHTreeNode *zero, *one;
    // Primaitives stored in the leaf
    int start_idx = 0, num_hitables = 0;
    // Axis the interior node was split along
    int split_axis = -1;
};

struct SAHBucket {
    int count = 0;
    Bounds bbox;
};

// BVHNode defined in bvh.h

enum PartitionMethod {
    Midpoint,
    SAH
};

using ObjectIt = std::vector<Object>::iterator;

/* Construct a BVH given a vector of objects and a partition method 
 * Modifies the objects vector such that every subtree covers a contiguous range */
std::vector<BVHNode> build_bvh(std::vector<Object>& objects, PartitionMethod method, Transform *transforms = nullptr);

/* Construct a binary tree by recursively partitioning across the midpoint of the object range */
BVHTreeNode *partition_midpoint(std::vector<Object>& objects, int start_idx, int count, const int leaf_size, Transform *transforms);

/* Construct a binary tree that minimizes the surface area heuristic */
BVHTreeNode *partition_sah(std::vector<Object>& objects, int start_idx, int count, const int leaf_size, Transform *transforms);

/* Takes a pointer-linked binary tree and flattens to linear storage */
void flatten_tree(std::vector<BVHNode>& bvh, BVHTreeNode *tree_node, int *idx);
void deallocate_tree(BVHTreeNode *root);
