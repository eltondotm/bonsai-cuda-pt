
#include <iostream>
#include <set>

#include <cuda_runtime.h>

#include <glm/vec3.hpp>
#include <miniScene/Scene.h>

#include "util/cuda_errors.h"
#include "util/random_declarations.h"
#include "util/image.h"
#include "util/file.h"
#include "scene.cpp"

//#define CPU
#define GPU

void render    (int sx, int sy, int ns, glm::vec3 *out, Scene scene);
void render_cpu(int sx, int sy, int ns, glm::vec3 *out, Scene scene);

int main(void) {
    int nx = 720;
    int ny = 720;
    int ns = 32768;
    int nchannels = 3;

    clock_t t = clock();
    std::cout << "Building BVH... ";
    Scene scene = create_scene("assets/cbox.mini");
    t = clock() - t;
    std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

    // Output buffer setup
    glm::vec3 *out;
    checkCudaErrors(cudaMallocManaged((void **)&out, nx*ny*sizeof(glm::vec3)));

    // Camera setup
    Camera *cam;
    checkCudaErrors(cudaMallocManaged((void **)&cam, sizeof(Camera)));
    *cam = Camera();
    cam->set_fov(45.f);
    cam->set_aspect((float)nx/ny);
    cam->look_at(glm::vec3(0.f, 1.f, 3.5f), glm::vec3(0.f, 1.f, 0.f));
    
    scene.camera = cam;

    // Non-accelerated list for testing
    #if 0
    Object *lst;
    checkCudaErrors(cudaMallocManaged((void **)&lst, n_bytes_obj));
    checkCudaErrors(cudaMemcpy(lst, prm_vec.data(), n_bytes_obj, cudaMemcpyHostToDevice));
    #endif

    #ifdef CPU
    {
        rng::init_host();
        clock_t t = clock();
        std::cout << "Rendering on CPU... ";
        render_cpu(nx, ny, ns, out, scene);
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        char *png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to cpu.png\n";
        write_png(write_filepath("cpu.png").c_str(), nx, ny, png);
        delete[] png;
        rng::cleanup_host();
    }
    #endif

    #ifdef GPU 
    {
        rng::init_device(nx, ny);
        clock_t t = clock();
        std::cout << "Rendering on GPU... ";
        render(nx, ny, ns, out, scene);
        checkCudaErrors(cudaDeviceSynchronize());
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        char *png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to gpu.png\n";
        write_png(write_filepath("gpu.png").c_str(), nx, ny, png);
        delete[] png;
        rng::cleanup_device();
    }
    #endif

    checkCudaErrors(cudaFree(out));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(scene.geometry));
    // checkCudaErrors(cudaFree(prm));
    // checkCudaErrors(cudaFree(bvh));
    // checkCudaErrors(cudaFree(lst));
    // checkCudaErrors(cudaFree(obj));
}
