
#include <iostream>
#include <map>

#include <cuda_runtime.h>

#include <glm/vec3.hpp>
#include <miniScene/Scene.h>

#include "util/cuda_errors.h"
#include "util/random.h"
#include "util/image.h"
#include "util/file.h"
#include "scene.h"
#include "mitsuba_scene.h"

void render    (int sx, int sy, int ns, glm::vec3 *out, Scene scene);
void render_cpu(int sx, int sy, int ns, glm::vec3 *out, Scene scene);
void render_speculative(int sx, int sy, int ns, glm::vec3 *out, Scene scene);
void render_persistent (int sx, int sy, int ns, glm::vec3 *out, Scene scene);

struct CameraParams {
    glm::vec3 pos;
    glm::vec3 target;
};

// Since obj files don't have camera position, save position for different files
static const std::map<std::string, CameraParams> camera_defaults{
    { "assets/cbox.mini", {glm::vec3(0.f, 1.f, 3.5f), glm::vec3(0.f, 1.f, 0.f)} },
    { "assets/cbox-water.mini", {glm::vec3(0.f, 0.75f, 3.5f), glm::vec3(0.f, 0.75f, 0.f)} },
    { "assets/dragon.mini", {glm::vec3(-0.5f, 10.f, 17.f), glm::vec3(-0.5f, 5.f, 0.f)} }
};

int main(int argc, char *argv[]) {
    int nx = 720;
    int ny = 720;
    int ns = 256;
    int nchannels = 3;
    char *scene_path = "assets/cbox.mini";
    char *output_path = "out.png";
    bool use_gpu = true;
    CameraParams cam_params{glm::vec3(0.f, 1.f, 3.5f), glm::vec3(0.f, 1.f, 0.f)};

    for (int i = 1; i != argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--width") {
            if (i + 1 < argc)
                nx = std::stoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--height") {
            if (i + 1 < argc)
                ny = std::stoi(argv[++i]);
        }
        else if (arg == "-s" || arg == "--samples") {
            if (i + 1 < argc)
                ns = std::stoi(argv[++i]);
        }
        else if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc)
                scene_path = argv[++i];
        }
        else if (arg == "-o" || arg == "--out") {
            if (i + 1 < argc)
                output_path = argv[++i];
        }
        else if (arg == "--cpu") {
            use_gpu = false;
        }
        else if (arg == "--gpu") {
            use_gpu = true;
        }
        else {
            std::cout << "Usage:\n"
                         "-w [int] or --width [int] sets output width\n"
                         "-h [int] or --height [int] sets output height\n"
                         "-s [int] or --samples [int] sets number of samples\n"
                         "-f [path] or --file [path] locates the input .mini file"
                         " relative to the root directory (or a parent).\n"
                         "-o [path] or --out [path] for the output .png file"
                         " relative to the root directory.\n"
                         "--cpu and --gpu specify rendering device.\n";
            return 0;
        }
    }

    clock_t t = clock();
    std::cout << "Building BVH... ";
    Scene scene = create_mitsuba_scene(scene_path, nx, ny);
    //Scene scene = create_scene(scene_path);
    t = clock() - t;
    std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

    // Output buffer setup
    glm::vec3 *out;
    checkCudaErrors(cudaMallocManaged((void **)&out, nx*ny*sizeof(glm::vec3)));

    // Camera setup
    if (camera_defaults.find(scene_path) != camera_defaults.end())
        cam_params = camera_defaults.at(scene_path);

    // Camera *cam;
    // checkCudaErrors(cudaMallocManaged((void **)&cam, sizeof(Camera)));
    // *cam = Camera();
    // cam->set_fov(45.f);
    // cam->set_aspect((float)nx/ny);
    // cam->look_at(cam_params.pos, cam_params.target);
    
    // scene.camera = cam;

    scene.camera->set_aspect((float)nx/ny);

    // Non-accelerated list for testing
    #if 0
    Object *lst;
    checkCudaErrors(cudaMallocManaged((void **)&lst, n_bytes_obj));
    checkCudaErrors(cudaMemcpy(lst, prm_vec.data(), n_bytes_obj, cudaMemcpyHostToDevice));
    #endif

    if (use_gpu) {
        rng::init_device(nx, ny);
        checkCudaErrors(cudaDeviceSynchronize());
        clock_t t = clock();
        std::cout << "Rendering on GPU with while-while... ";
        render(nx, ny, 1, out, scene);
        checkCudaErrors(cudaDeviceSynchronize());
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        char *png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to " << output_path << std::endl;
        write_png(write_filepath(output_path).c_str(), nx, ny, png);
        delete[] png;

        rng::init_device(nx, ny);
        checkCudaErrors(cudaMemset(out, 0, nx*ny*sizeof(glm::vec3)));
        t = clock();
        std::cout << "Rendering on GPU with persistent threads... ";
        render_persistent(nx, ny, ns, out, scene);
        checkCudaErrors(cudaDeviceSynchronize());
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to persistent.png" << std::endl;
        write_png(write_filepath("persistent.png").c_str(), nx, ny, png);
        delete[] png;
        rng::cleanup_device();
    }

    if (!use_gpu) {
        rng::init_host();
        clock_t t = clock();
        std::cout << "Rendering on CPU... ";
        render_cpu(nx, ny, ns, out, scene);
        t = clock() - t;
        std::cout << "took " << (double)t/CLOCKS_PER_SEC << " seconds\n";

        char *png = vec_to_byte(out, nx, ny);
        std::cout << "Writing to " << output_path << std::endl;
        write_png(write_filepath(output_path).c_str(), nx, ny, png);
        delete[] png;
        rng::cleanup_host();
    }

    free_scene(scene);
    checkCudaErrors(cudaFree(out));
    return 0;
}
