
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

#define SM_COUNT 48
#define WARP_SIZE 32
#define WARPS_PER_SM 48
#define BLOCK_SIZE 256
#define BLOCKDIM_X WARP_SIZE
#define BLOCKDIM_Y BLOCK_SIZE / BLOCKDIM_X

#define BATCH_SIZE WARP_SIZE * 4

#define FULL_MASK 0xFFFFFFFF

__constant__ int d_queue_length;
__device__   int d_queue_index = 0;

__global__ void d_init_work_queue(int sx, int sy, RayData *queue, Scene scene) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sx || j >= sy) return;
    int pixel_index = j*sx + i;

    glm::vec2 coords((float)i,  (float)j );
    glm::vec2 dims  ((float)sx, (float)sy);
    
    glm::vec2 uv = (coords + rng::square()) / dims;
    Ray r = scene.camera->generate_ray(uv);
    
    queue[pixel_index] = RayData(r, pixel_index);
}

// Test to make sure the ray queue is functional
__global__ void d_render_persistent_debug(int sx, int sy, RayData *queue, glm::vec3 *out, Scene scene) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sx || j >= sy) return;
    int pixel_index = j*sx + i;

    RayData& rd = queue[pixel_index];
    while (rd.bounces != -1) {
        trace(rd, scene);
        /* Print debugging
        if (i == 300 && j == 300) {
            glm::vec3 val = rd.radiance;
            printf("%f, %f, %f, %d\n", val.r, val.g, val.b, rd.bounces);
        } */
    }
    out[pixel_index] += queue[pixel_index].radiance;
}

// Adapted from Aila and Laine
__global__ void d_render_persistent(RayData *queue, glm::vec3 *out, Scene scene) {
    // Arrays for full block, one value per warp
    __shared__ volatile int next_ray_arr[BLOCKDIM_Y];
    __shared__ volatile int ray_count_arr[BLOCKDIM_Y];
    if (threadIdx.x == 0)
        ray_count_arr[threadIdx.y] = 0;


    volatile int& local_next_ray = next_ray_arr[threadIdx.y];
    volatile int& local_ray_count = ray_count_arr[threadIdx.y];

    while (true) {
        // Fetch new rays if local rays depleted
        if (local_ray_count == 0 && threadIdx.x == 0) {
            local_next_ray = atomicAdd(&d_queue_index, BATCH_SIZE);
            local_ray_count = BATCH_SIZE;
        }
        // Fetch next local ray
        int ray_index = (local_next_ray + threadIdx.x) % d_queue_length;

        // Return if the entire batch is already terminated
        // Ideally, some sorting of the queue would ensure this only happens at the very end
        RayData& rd = queue[ray_index];

        if (__all_sync(FULL_MASK, rd.bounces == -1))
            return;

        if (threadIdx.x == 0) {
            local_next_ray += WARP_SIZE;
            local_ray_count -= WARP_SIZE;
        }

        // Trace a single bounce and update ray state
        if (rd.bounces != -1) {
            trace(queue[ray_index], scene);
            if (rd.bounces == -1) {
                out[rd.pixel_index] += rd.radiance;
            }  
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
        d_init_work_queue<<<img_blocks, img_threads>>>(sx, sy, queue, scene);
        checkCudaErrors(cudaDeviceSynchronize());

        // Launching enought threads for full occupancy
        dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
        dim3 blocks(SM_COUNT * WARPS_PER_SM * WARP_SIZE / BLOCK_SIZE);
        d_render_persistent<<<blocks, threads>>>(queue, out, scene);
        //d_render_persistent_debug<<<img_blocks, img_threads>>>(sx, sy, queue, out, scene);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    d_normalize_color<<<img_blocks, img_threads>>>(sx, sy, ns, out, queue);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(queue));
}
