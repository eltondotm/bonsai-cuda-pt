
#include "bvh_build.h"

#include <algorithm>
#include <limits>

std::vector<BVHNode> build_bvh(std::vector<Object>& objects, PartitionMethod method) {
    BVHTreeNode *tree;
    switch (method) {
        case Midpoint:
            tree = partition_midpoint(objects, 0, (int)objects.size(), 1);
            break;
        case SAH:
        default:
            tree = partition_sah(objects, 0, (int)objects.size(), 1);
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

BVHTreeNode *partition_sah(std::vector<Object>& objects, int start_idx, int count, const int leaf_size) {
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


    // Construct bounding boxes: centroid for split, combined for surface area calculation
    Bounds centroid_bounds;
    Bounds combined_bounds;
    for (ObjectIt h = start; h != end; ++h) {
        centroid_bounds.enclose(h->bounds().centroid());
        combined_bounds.enclose(h->bounds());
    }
    // Find the midpoint along the largest dimension
    int dim = centroid_bounds.max_extent_dim();

    constexpr int n_buckets = 12;
    
    // Bucketing logic references PBRT
    SAHBucket buckets[n_buckets];
    float costs[n_buckets-1];

    for (const Object& obj : objects) {
        int bucket = n_buckets * centroid_bounds.normalize_point(obj.bounds().centroid())[dim];
        if (bucket == n_buckets)
            bucket = n_buckets-1;
        buckets[bucket].count++;
        buckets[bucket].bbox.enclose(obj.bounds());
    }

    // Add the cost of the left child to costs
    int count_left = 0;
    Bounds bound_left;
    for (int i = 0; i < n_buckets - 1; ++i) {
        bound_left.enclose(buckets[i].bbox);
        count_left += buckets[i].count;
        costs[i] = count_left * bound_left.surface_area();
    }

    // Add the cost of the right child to costs
    int count_above = 0;
    Bounds bound_right;
    for (int i = n_buckets - 1; i > 0; --i) {
        bound_right.enclose(buckets[i].bbox);
        count_above += buckets[i].count;
        costs[i - 1] += count_above * bound_right.surface_area();
    }

    // Find the best split
    int best_bucket = -1;
    float min_cost = std::numeric_limits<float>::max();
    for (int i = 0; i < n_buckets - 1; ++i) {
        // Compute cost for candidate split and update minimum if
        // necessary
        if (costs[i] < min_cost) {
            min_cost = costs[i];
            best_bucket = i;
        }
    }

    // c_trav = 0.5, c_isect = 1.0
    min_cost = 0.5f + min_cost / combined_bounds.surface_area();

    // Partition based on centroid dimension
    ObjectIt split_it = std::partition(start, end, 
        [dim, n_buckets, best_bucket, &centroid_bounds](const Object& h) {
            int bucket = n_buckets * centroid_bounds.normalize_point(h.bounds().centroid())[dim];
            return bucket <= best_bucket;
        });

    // Fallback: divide in half if SAH fails to split
    if (split_it == start || split_it == end) {
        split_it = start + (count / 2);
        std::nth_element(start, split_it, end, 
            [dim](const Object& a, const Object& b) {
                return a.bounds().centroid()[dim] < b.bounds().centroid()[dim];
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
    root->create_interior(zero, one, dim);
    return root;
}

// Pack binary tree into a contiguous array
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
