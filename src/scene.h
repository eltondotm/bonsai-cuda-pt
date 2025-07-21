
#pragma once

#include "object.h"
#include "camera.h"

struct Scene {
    BVH<Object> *geometry = nullptr;
    Camera *camera = nullptr;
    Object **emitters = nullptr;
    int n_emitters = 0;
};

// Util functions for converting between miniScene and glm
inline glm::vec3 mini_to_vec3(mini::vec3f mini);
inline glm::ivec3 mini_to_ivec3(mini::vec3i mini);

// Converting mini::DisneyMaterial to basic materials (cutoffs are arbitrary)
Material mini_disney_to_material(mini::DisneyMaterial::SP mini_mat);

// Loading a .mini file, then converting to a gpu-friendly format for rendering
Scene create_scene(const char *filename);

// Deallocates a BVH of triangle mesh BVHs
void free_tri_mesh_bvh(const BVH<Object> *geo, int idx);

// Assumes a two level BVH (which is true for any scene built with create_scene())
void free_scene(const Scene& scene);
