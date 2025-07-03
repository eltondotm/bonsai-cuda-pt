
#include <iostream>

#include <cuda_runtime.h>

#include <glm/vec3.hpp>

#include "util/cuda_errors.h"
#include "util/image.h"
#include "camera.h"
#include "object.h"
#include "bvh_build.h"

#define GPU
#define CPU

void render    (int sx, int sy, glm::vec3 *out, Camera *cam, BVH<Object> *obj);
void render_cpu(int sx, int sy, glm::vec3 *out, Camera *cam, BVH<Object> *obj);

int main(void) {
    int nx = 1280;
    int ny = 720;
    int nchannels = 3;

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
    std::vector<Object> prm_vec;
    for (int j = 0; j < 128; ++j) {
        float z = 7.5f - (float)j;
        for (int i = 0; i < 32; ++i) {
            float x = 15.5f - (float)i;
            prm_vec.push_back(Object(Sphere(0.5f, glm::vec3(x, 0.f, z))));
        }
    }

    std::vector<BVHNode> bvh_vec = build_bvh(prm_vec, Midpoint);

    // Convert to arrays to send to device
    int n_bytes_obj = prm_vec.size()*sizeof(Object);
    int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);

    Object *prm;
    checkCudaErrors(cudaMallocManaged((void **)&prm, n_bytes_obj));
    checkCudaErrors(cudaMemcpy(prm, prm_vec.data(), n_bytes_obj, cudaMemcpyHostToDevice));

    BVHNode *bvh;
    checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    // Non-accelerated list for testing
    Object *lst;
    checkCudaErrors(cudaMallocManaged((void **)&lst, n_bytes_obj));
    checkCudaErrors(cudaMemcpy(lst, prm_vec.data(), n_bytes_obj, cudaMemcpyHostToDevice));

    BVH<Object> *obj;
    checkCudaErrors(cudaMallocManaged((void **)&obj, sizeof(BVH<Object>)));
    *obj = BVH<Object>(prm, prm_vec.size(), bvh);

    #ifdef GPU 
    {
        // Band-aid fix -- recursion caused stack overflow
        //cudaDeviceSetLimit(cudaLimitStackSize, 4096);

        clock_t t = clock();
        std::cout << "Rendering on GPU... ";
        render(nx, ny, out, cam, obj);
        checkCudaErrors(cudaDeviceSynchronize());
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        char *png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to gpu.png\n";
        write_png("gpu.png", nx, ny, png);
        delete[] png;
    }
    #endif

    #ifdef CPU
    {
        clock_t t = clock();
        std::cout << "Rendering on CPU... ";
        render_cpu(nx, ny, out, cam, obj);
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        char *png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to cpu.png\n";
        write_png("cpu.png", nx, ny, png);
        delete[] png;
    }
    #endif

    checkCudaErrors(cudaFree(out));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(prm));
    checkCudaErrors(cudaFree(bvh));
    checkCudaErrors(cudaFree(lst));
    checkCudaErrors(cudaFree(obj));
}
