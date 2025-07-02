
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
    int start_idx, num_hitables;
    // Axis the interior node was split along
    int split_axis;
};

// BVHNode defined in bvh.h

enum PartitionMethod {
    Midpoint,
    SAH
};

using ObjectIt = std::vector<Object>::iterator;

std::vector<BVHNode> build_bvh(std::vector<Object>& objects, PartitionMethod method);
BVHTreeNode *partition_midpoint(std::vector<Object>& objects, int start_idx, int count, const int leaf_size);
void flatten_tree(std::vector<BVHNode>& bvh, BVHTreeNode *tree_node, int *idx);
void deallocate_tree(BVHTreeNode *root);
