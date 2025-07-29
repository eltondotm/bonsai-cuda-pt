
#include <cooperative_groups.h>

#include <glm/vec3.hpp>

#include "util/cuda_errors.h"
#include "util/random.h"
#include "scene.h"
#include "trace.h"

inline __device__ void saturate(glm::vec3& color) {
    color.r = __saturatef(color.r);
    color.g = __saturatef(color.g);
    color.b = __saturatef(color.b);
}

/////////////////////////////
// DEFAULT RENDER ALGORITHM
/////////////////////////////

__global__ void d_render(int sx, int sy, int ns, glm::vec3 *out, Scene scene) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sx || j >= sy) return;
    int pixel_index = j*sx + i;

    glm::vec2 coords((float)i,  (float)j );
    glm::vec2 dims  ((float)sx, (float)sy);

    glm::vec3 color(0.f);
    for (int i = 0; i < ns; ++i) {
        glm::vec2 uv = (coords + rng::square()) / dims;
        Ray r = scene.camera->generate_ray(uv);
        color += trace_ray(r, scene);
    }
    color /= (float)ns;
    color = glm::sqrt(color);
    saturate(color);

    out[pixel_index] = color;
}

void render(int sx, int sy, int ns, glm::vec3 *out, Scene scene) {
    dim3 threads(16, 16);
    dim3 blocks((sx+threads.x-1) / threads.x, (sy+threads.y-1) / threads.y);
    d_render<<<blocks, threads>>>(sx, sy, ns, out, scene);
}

__global__ void d_render_speculative(int sx, int sy, int ns, glm::vec3 *out, Scene scene) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sx || j >= sy) return;
    int pixel_index = j*sx + i;

    glm::vec2 coords((float)i,  (float)j );
    glm::vec2 dims  ((float)sx, (float)sy);

    glm::vec3 color(0.f);
    for (int i = 0; i < ns; ++i) {
        glm::vec2 uv = (coords + rng::square()) / dims;
        Ray r = scene.camera->generate_ray(uv);
        color += trace_speculative(r, scene);
    }
    color /= (float)ns;
    color = glm::sqrt(color);
    saturate(color);

    out[pixel_index] = color;
}

void render_speculative(int sx, int sy, int ns, glm::vec3 *out, Scene scene) {
    dim3 threads(16, 16);
    dim3 blocks((sx+threads.x-1) / threads.x, (sy+threads.y-1) / threads.y);
    d_render_speculative<<<blocks, threads>>>(sx, sy, ns, out, scene);
}


/////////////////////////////
// PERSISTENT THREADS
/////////////////////////////

#define SM_COUNT 46
#define WARP_SIZE 32
#define WARPS_PER_SM 48
#define BLOCK_SIZE 256
#define BLOCKDIM_X WARP_SIZE
#define BLOCKDIM_Y BLOCK_SIZE / BLOCKDIM_X

#define BATCH_SIZE WARP_SIZE * 4

#define FULL_MASK 0xFFFFFFFF

using cooperative_groups::details::lanemask32_lt;

__constant__ int d_queue_length;
__device__   int d_queue_index = 0;

__global__ void d_init_work_queue(int sx, int sy, RayData *queue, Scene scene) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sx || j >= sy) return;
    int pixel_index = j*sx + i;

    if (i == 0 && j == 0) d_queue_index = 0;

    glm::vec2 coords((float)i,  (float)j );
    glm::vec2 dims  ((float)sx, (float)sy);
    
    glm::vec2 uv = (coords + rng::square()) / dims;
    Ray r = scene.camera->generate_ray(uv);
    
    queue[pixel_index] = RayData(r, pixel_index);
}

inline __device__ RayData init_ray(int w, int h, int index, Camera *cam) {
    int x = index % w;
    int y = index / w;

    glm::vec2 coords((float)x, (float)y);
    glm::vec2 dims  ((float)w, (float)h);

    glm::vec2 uv = (coords + rng::square()) / dims;
    Ray r = cam->generate_ray(uv);

    return RayData(r, index);
}

__global__ void d_render_persistent(int w, int h, glm::vec3 *out, Scene scene) {
    // Reset global counter
    if (blockIdx.x == 0 && threadIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0)
        d_queue_index = 0;
    
    int i;  // Current ray index

    // Fetch initial rays
    unsigned int mask = __activemask();
    unsigned int rays_needed = __popc(mask);
    unsigned int rank = __popc(mask & lanemask32_lt());
    int representative = __ffs(mask) - 1;
    if (rank == 0) {
        i = atomicAdd(&d_queue_index, rays_needed);
    }
    i = __shfl_sync(mask, i, representative) + rank;

    RayData rd = init_ray(w, h, i, scene.camera);
    while (i < d_queue_length) {
        trace(rd, scene);
        if (rd.bounces == -1) {
            out[rd.pixel_index] += rd.radiance;
        }

        // Fetch more rays if needed
        unsigned int terminated = __ballot_sync(__activemask(), rd.bounces == -1);
        if (rd.bounces == -1) {
            unsigned int rank = __popc(terminated & lanemask32_lt());
            int rays_needed = __popc(terminated);
            int representative = __ffs(terminated) - 1;
            if (rank == 0)
                i = atomicAdd(&d_queue_index, rays_needed);
            i = __shfl_sync(terminated, i, representative) + rank;
            rd = init_ray(w, h, i, scene.camera);
        }
    }
}

__global__ void d_normalize_color(int sx, int sy, int ns, glm::vec3 *out, RayData *queue) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sx || j >= sy) return;
    int pixel_index = j*sx + i;

    glm::vec3& color = out[pixel_index];
    color /= (float)ns;
    color = glm::sqrt(color);
    saturate(color);
}

void render_persistent(int sx, int sy, int ns, glm::vec3 *out, Scene scene) {
    // Creating global work queue
    RayData *queue;
    int queue_length = sx*sy;
    checkCudaErrors(cudaMalloc((void **)&queue, queue_length*sizeof(RayData)));
    checkCudaErrors(cudaMemcpyToSymbol(d_queue_length, &queue_length, sizeof(int)));

    dim3 img_threads(16, 16);
    dim3 img_blocks((sx+img_threads.x-1) / img_threads.x, (sy+img_threads.y-1) / img_threads.y);
    for (int i = 0; i < ns; ++i) {
        // Launching enought threads for full occupancy
        dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
        dim3 blocks(SM_COUNT * WARPS_PER_SM * WARP_SIZE / BLOCK_SIZE);
        d_render_persistent<<<blocks, threads>>>(sx, sy, out, scene);
        //d_render_persistent_debug<<<img_blocks, img_threads>>>(sx, sy, queue, out, scene);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    d_normalize_color<<<img_blocks, img_threads>>>(sx, sy, ns, out, queue);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(queue));
}
