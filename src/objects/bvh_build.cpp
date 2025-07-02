
#include "bvh_build.h"

#include <algorithm>

std::vector<BVHNode> build_bvh(std::vector<Object>& objects, PartitionMethod method) {
    BVHTreeNode *tree;
    switch (method) {
        case SAH:
        case Midpoint:
        default:
            tree = partition_midpoint(objects, 0, (int)objects.size(), 1);
            break;
    }

    std::vector<BVHNode> bvh;
    int idx = 0;
    flatten_tree(bvh, tree, &idx);
    deallocate_tree(tree);

    return bvh;
}

// Divide objects along bounding box midpoint
BVHTreeNode *partition_midpoint(std::vector<Object>& objects, int start_idx, int count, const int leaf_size) {
    // Get iterators to the bounds of the range being sorted
    ObjectIt start = objects.begin() + start_idx;
    ObjectIt end   = start + count;
    
    // Base case: leaf node
    if (count <= leaf_size) {
        Bounds leaf_bounds;
        for (ObjectIt h = start; h != end; ++h) {
            leaf_bounds = union_bounds(leaf_bounds, h->bounds());
        }
        BVHTreeNode *leaf = new BVHTreeNode;
        leaf->create_leaf(leaf_bounds, start_idx, count);
        return leaf;
    }

    // Construct bounding box for centroids of all boxes in the range
    Bounds centroid_bounds;
    for (ObjectIt h = start; h != end; ++h) {
        centroid_bounds.enclose(h->bounds().centroid());
    }
    // Find the midpoint along the largest dimension
    int split_dim = centroid_bounds.max_extent_dim();
    float midpoint = centroid_bounds.centroid()[split_dim];

    // Partition based on centroid dimension
    ObjectIt split_it = std::partition(start, end, 
        [split_dim, midpoint](const Object& h) {
            return h.bounds().centroid()[split_dim] < midpoint;
        });

    // Fallback: divide in half if midpoint fails to split
    if (split_it == start || split_it == end) {
        split_it = start + (count / 2);
        std::nth_element(start, split_it, end, 
            [split_dim](const Object& a, const Object& b) {
                return a.bounds().centroid()[split_dim] < b.bounds().centroid()[split_dim];
            });
    }

    // Recursively partition and return the root node
    int start_zero = start_idx;
    int num_zero   = (int)(split_it - start);
    int start_one  = start_idx + num_zero;
    int num_one    = count - num_zero;
    BVHTreeNode *zero = partition_midpoint(objects, start_zero, num_zero, leaf_size);
    BVHTreeNode *one  = partition_midpoint(objects, start_one,  num_one,  leaf_size);
    BVHTreeNode *root = new BVHTreeNode;
    root->create_interior(zero, one, split_dim);
    return root;
}

// BVH array must be already allocated
void flatten_tree(std::vector<BVHNode>& bvh, BVHTreeNode *tree_node, int *idx) {
    int curr_idx = *idx;
    ++(*idx);  // Move to next array position
    bvh.push_back(BVHNode());  // Placeholder to be filled in

    // Initialize next linear tree node
    BVHNode lin_node;
    lin_node.bbox = tree_node->bbox;

    // Leaf node
    if (tree_node->num_hitables > 0) {
        lin_node.start_idx = tree_node->start_idx;
        lin_node.num_hitables = tree_node->num_hitables;
        bvh[curr_idx] = lin_node;
        return;
    }

    // Recursively fill in subtrees to get one_idx for interior node
    flatten_tree(bvh, tree_node->zero, idx);
    int one_idx = *idx;
    flatten_tree(bvh, tree_node->one,  idx);

    // Interior node
    lin_node.axis = tree_node->split_axis;
    lin_node.num_hitables = 0;
    lin_node.one_idx = one_idx;
    bvh[curr_idx] = lin_node;   
}

void deallocate_tree(BVHTreeNode *root) {
    if (root->zero != nullptr) {
        deallocate_tree(root->zero);
        deallocate_tree(root->one);
    }
    delete root;
}
