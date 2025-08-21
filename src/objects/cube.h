
#include <cuda_runtime.h>

#include "util/cuda_errors.h"
#include "bvh_build.h"
#include "object.h"

BVH<Object> construct_cube(uint16_t mat_idx, uint16_t trans, Transform *transforms) {
    constexpr glm::vec3 vertices[24] = {
        {  1, -1, -1 }, {  1, -1,  1 }, { -1, -1,  1 }, { -1, -1, -1 },
        {  1,  1, -1 }, { -1,  1, -1 }, { -1,  1,  1 }, {  1,  1,  1 },
        {  1, -1, -1 }, {  1,  1, -1 }, {  1,  1,  1 }, {  1, -1,  1 },
        {  1, -1,  1 }, {  1,  1,  1 }, { -1,  1,  1 }, { -1, -1,  1 },
        { -1, -1,  1 }, { -1,  1,  1 }, { -1,  1, -1 }, { -1, -1, -1 },
        {  1,  1, -1 }, {  1, -1, -1 }, { -1, -1, -1 }, { -1,  1, -1 }
    };
    constexpr glm::vec3 normals[24] = {
        { 0, -1,  0 }, {  0, -1,  0 }, {  0, -1,  0 }, {  0, -1,  0 }, {  0, 1, 0 },
        { 0,  1,  0 }, {  0,  1,  0 }, {  0,  1,  0 }, {  1,  0,  0 }, {  1, 0, 0 },
        { 1,  0,  0 }, {  1,  0,  0 }, {  0,  0,  1 }, {  0,  0,  1 }, {  0, 0, 1 },
        { 0,  0,  1 }, { -1,  0,  0 }, { -1,  0,  0 }, { -1,  0,  0 }, { -1, 0, 0 },
        { 0,  0, -1 }, {  0,  0, -1 }, {  0,  0, -1 }, {  0,  0, -1 }
    };
    constexpr glm::ivec3 triangles[12] = {
        {  0,  1,  2 }, {  3,  0,  2 }, {  4,  5,  6 }, {  7,  4,  6 },
        {  8,  9, 10 }, { 11,  8, 10 }, { 12, 13, 14 }, { 15, 12, 14 },
        { 16, 17, 18 }, { 19, 16, 18 }, { 20, 21, 22 }, { 23, 20, 22 }
    };

    glm::vec3 *verts;
    glm::vec3 *norms;
    int v_size = 24*sizeof(glm::vec3);
    int n_size = 24*sizeof(glm::vec3);
    checkCudaErrors(cudaMallocManaged((void **)&verts, v_size));
    checkCudaErrors(cudaMallocManaged((void **)&norms, n_size));
    checkCudaErrors(cudaMemcpy(verts, &vertices, v_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(norms, &normals,  n_size, cudaMemcpyHostToDevice));

    TriangleMesh *tri_mesh;
    checkCudaErrors(cudaMallocManaged((void**)&tri_mesh, sizeof(TriangleMesh)));
    *tri_mesh = TriangleMesh{verts, norms, nullptr, nullptr, true};

    std::vector<Object> tri_vec;
    for (int i = 0; i < 12; ++i) {
        tri_vec.push_back(Object(Triangle(tri_mesh, triangles[i]), mat_idx, trans));
    }

    std::vector<BVHNode> bvh_vec = build_bvh(tri_vec, SAH, transforms);

    int n_bytes_tri = tri_vec.size()*sizeof(Object);
    int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);

    Object *tri;
    checkCudaErrors(cudaMallocManaged((void **)&tri, n_bytes_tri));
    checkCudaErrors(cudaMemcpy(tri, tri_vec.data(), n_bytes_tri, cudaMemcpyHostToDevice));

    BVHNode *bvh;
    checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    return BVH<Object>(tri, tri_vec.size(), bvh, bvh_vec.size());
}

BVH<Object> construct_square(uint16_t mat_idx, uint16_t trans, Transform *transforms) {
    glm::vec3 vertices[4] = {
        { 1,  -1,  0 }, { 1,  1,  0 }, { -1,  1,  0 }, { -1, -1,  0 }
    };
    glm::vec3 normals[4] = {
        { 0, 0, 1 }, { 0, 0, 1 }, { 0, 0, 1 }, { 0, 0, 1 }
    };
    glm::ivec3 triangles[2] = {
        {  0,  1,  2 }, {  3,  0,  2 }
    };

    for (int i = 0; i < 4; ++i) {
        glm::vec3& vi = vertices[i];
        glm::vec3& ni = normals[i];
        vi = transforms[trans].trans * glm::vec4(vi, 1.f);
        ni = transforms[trans].trans * glm::vec4(ni, 0.f);
        // vi = glm::vec3(vi.x, vi.z, vi.y);
        // ni = glm::vec3(ni.x, ni.z, ni.y);
    }
    trans = 0;

    glm::vec3 *verts;
    glm::vec3 *norms;
    int v_size = 4*sizeof(glm::vec3);
    int n_size = 4*sizeof(glm::vec3);
    checkCudaErrors(cudaMallocManaged((void **)&verts, v_size));
    checkCudaErrors(cudaMallocManaged((void **)&norms, n_size));
    checkCudaErrors(cudaMemcpy(verts, &vertices, v_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(norms, &normals,  n_size, cudaMemcpyHostToDevice));

    TriangleMesh *tri_mesh;
    checkCudaErrors(cudaMallocManaged((void**)&tri_mesh, sizeof(TriangleMesh)));
    *tri_mesh = TriangleMesh{verts, norms, nullptr, nullptr, true};

    std::vector<Object> tri_vec;
    for (int i = 0; i < 2; ++i) {
        tri_vec.push_back(Object(Triangle(tri_mesh, triangles[i]), mat_idx, trans));
    }

    std::vector<BVHNode> bvh_vec = build_bvh(tri_vec, SAH, transforms);

    int n_bytes_tri = tri_vec.size()*sizeof(Object);
    int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);

    Object *tri;
    checkCudaErrors(cudaMallocManaged((void **)&tri, n_bytes_tri));
    checkCudaErrors(cudaMemcpy(tri, tri_vec.data(), n_bytes_tri, cudaMemcpyHostToDevice));

    BVHNode *bvh;
    checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    return BVH<Object>(tri, tri_vec.size(), bvh, bvh_vec.size());
}
