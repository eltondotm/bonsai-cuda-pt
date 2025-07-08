
#include <iostream>
#include <set>

#include <cuda_runtime.h>

#include <glm/vec3.hpp>
#include <miniScene/Scene.h>

#include "util/cuda_errors.h"
#include "util/image.h"
#include "util/file.h"
#include "camera.h"
#include "object.h"
#include "triangle.h"
#include "bvh_build.h"

#define GPU
#define CPU

void render    (int sx, int sy, glm::vec3 *out, Camera *cam, BVH<Object> *obj);
void render_cpu(int sx, int sy, glm::vec3 *out, Camera *cam, BVH<Object> *obj);

// Util functions for converting between miniScene and glm
inline glm::vec3 mini_to_vec3(mini::vec3f mini) { return glm::vec3(mini.x, mini.y, mini.z); }
inline glm::ivec3 mini_to_ivec3(mini::vec3i mini) { return glm::ivec3(mini.r, mini.s, mini.t); }

BVH<Object> *create_scene(const char *filename) {
    mini::Scene::SP scene = mini::Scene::load(read_filepath(filename));

    // Gathering unique meshes
    std::set<mini::Mesh::SP> meshes;
    for (const mini::Instance::SP inst : scene->instances) {
        for (const mini::Mesh::SP mesh : inst->object->meshes) {
            meshes.insert(mesh);
        }
    }

    std::vector<Object> tri_bvhs;
    for (const mini::Mesh::SP mesh : meshes) {
        // Moving mesh data to arrays for access from host or device
        glm::vec3 *verts;
        glm::vec3 *norms;
        int v_size = mesh->vertices.size()*sizeof(glm::vec3);
        int n_size = mesh->normals.size()*sizeof(glm::vec3);
        checkCudaErrors(cudaMallocManaged((void **)&verts, v_size));
        checkCudaErrors(cudaMallocManaged((void **)&norms, n_size));
        checkCudaErrors(cudaMemcpy(verts, mesh->vertices.data(), v_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(norms, mesh->normals.data(), n_size, cudaMemcpyHostToDevice));

        TriangleMesh tri_mesh{verts, norms, !mesh->normals.empty()};

        std::vector<Object> tri_vec;
        for (int i = 0; i < mesh->getNumPrims(); ++i) {
            tri_vec.push_back(Object(Triangle(&tri_mesh, mini_to_ivec3(mesh->indices[i]))));
        }

        // BVH construction
        std::vector<BVHNode> bvh_vec = build_bvh(tri_vec, Midpoint);

        int n_bytes_tri = tri_vec.size()*sizeof(Object);
        int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);

        Object *tri;
        checkCudaErrors(cudaMallocManaged((void **)&tri, n_bytes_tri));
        checkCudaErrors(cudaMemcpy(tri, tri_vec.data(), n_bytes_tri, cudaMemcpyHostToDevice));

        BVHNode *bvh;
        checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
        checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

        tri_bvhs.push_back(BVH<Object>(tri, tri_vec.size(), bvh));
    }

    std::vector<BVHNode> scn_bvh = build_bvh(tri_bvhs, Midpoint);

    int n_bytes_scn = tri_bvhs.size()*sizeof(Object);
    int n_bytes_bvh = scn_bvh.size()*sizeof(BVHNode);

    Object *scn;
    checkCudaErrors(cudaMallocManaged((void **)&scn, n_bytes_scn));
    checkCudaErrors(cudaMemcpy(scn, tri_bvhs.data(), n_bytes_scn, cudaMemcpyHostToDevice));

    BVHNode *bvh;
    checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(bvh, scn_bvh.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    BVH<Object> *obj;
    checkCudaErrors(cudaMallocManaged((void **)&obj, sizeof(BVH<Object>)));
    *obj = BVH<Object>(scn, tri_bvhs.size(), bvh);

    return obj;
}

int main(void) {
    int nx = 1280;
    int ny = 720;
    int nchannels = 3;

    clock_t t = clock();
    std::cout << "Building BVH... ";
    BVH<Object> *scn = create_scene("goat.mini");
    t = clock() - t;
    std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

    // Output buffer setup
    glm::vec3 *out;
    checkCudaErrors(cudaMallocManaged((void **)&out, nx*ny*sizeof(glm::vec3)));

    // Camera setup
    Camera *cam;
    checkCudaErrors(cudaMallocManaged((void **)&cam, sizeof(Camera)));
    *cam = Camera();
    cam->set_aspect((float)nx/ny);
    cam->look_at(glm::vec3(0.f, -5.f, -7.f), glm::vec3(0.f));

    // BVH setup
    // std::vector<Object> prm_vec;
    // for (int j = 0; j < 128; ++j) {
    //     float z = 7.5f - (float)j;
    //     for (int i = 0; i < 32; ++i) {
    //         float x = 15.5f - (float)i;
    //         prm_vec.push_back(Object(Sphere(0.5f, glm::vec3(x, 0.f, z))));
    //     }
    // }

    // std::vector<BVHNode> bvh_vec = build_bvh(prm_vec, Midpoint);

    // // Convert to arrays to send to device
    // int n_bytes_obj = prm_vec.size()*sizeof(Object);
    // int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);

    // Object *prm;
    // checkCudaErrors(cudaMallocManaged((void **)&prm, n_bytes_obj));
    // checkCudaErrors(cudaMemcpy(prm, prm_vec.data(), n_bytes_obj, cudaMemcpyHostToDevice));

    // BVHNode *bvh;
    // checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
    // checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    // BVH<Object> *obj;
    // checkCudaErrors(cudaMallocManaged((void **)&obj, sizeof(BVH<Object>)));
    // *obj = BVH<Object>(prm, prm_vec.size(), bvh);

    // Non-accelerated list for testing
    #if 0
    Object *lst;
    checkCudaErrors(cudaMallocManaged((void **)&lst, n_bytes_obj));
    checkCudaErrors(cudaMemcpy(lst, prm_vec.data(), n_bytes_obj, cudaMemcpyHostToDevice));
    #endif

    #ifdef CPU
    {
        clock_t t = clock();
        std::cout << "Rendering on CPU... ";
        render_cpu(nx, ny, out, cam, scn);
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        char *png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to cpu.png\n";
        write_png(write_filepath("cpu.png").c_str(), nx, ny, png);
        delete[] png;
    }
    #endif

    #ifdef GPU 
    {
        clock_t t = clock();
        std::cout << "Rendering on GPU... ";
        render(nx, ny, out, cam, scn);
        checkCudaErrors(cudaDeviceSynchronize());
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        char *png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to gpu.png\n";
        write_png(write_filepath("gpu.png").c_str(), nx, ny, png);
        delete[] png;
    }
    #endif

    checkCudaErrors(cudaFree(out));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(scn));
    // checkCudaErrors(cudaFree(prm));
    // checkCudaErrors(cudaFree(bvh));
    // checkCudaErrors(cudaFree(lst));
    // checkCudaErrors(cudaFree(obj));
}
