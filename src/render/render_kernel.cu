
#include <glm/vec3.hpp>

#include "camera.h"
#include "trace_ray.h"

__global__ void d_render(int sx, int sy, glm::vec3 *out, Camera *cam, BVH<Object> *obj) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sx || j >= sy) return;
    int pixel_index = j*sx + i;

    glm::vec2 uv((float)i/sx, (float)j/sy);
    Ray r = cam->generate_ray(uv);

    out[pixel_index] = trace_ray(r, obj);
}

void render(int sx, int sy, glm::vec3 *out, Camera *cam, BVH<Object> *obj) {
    dim3 threads(8, 8);
    dim3 blocks((sx+threads.x-1) / threads.x, (sy+threads.y-1) / threads.y);
    d_render<<<blocks, threads>>>(sx, sy, out, cam, obj);
}
