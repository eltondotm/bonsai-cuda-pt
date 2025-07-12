
#include <glm/vec3.hpp>

#include <util/random_declarations.h>
#include "camera.h"
#include "trace_ray.h"

__global__ void d_render(int sx, int sy, int ns, glm::vec3 *out, Camera *cam, BVH<Object> *obj) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= sx || j >= sy) return;
    int pixel_index = j*sx + i;

    glm::vec2 coords((float)i,  (float)j );
    glm::vec2 dims  ((float)sx, (float)sy);

    glm::vec3 color(0.f);
    for (int i = 0; i < ns; ++i) {
        glm::vec2 uv = (coords + rng::square()) / dims;
        Ray r = cam->generate_ray(uv);
        color += trace_ray(r, obj);
    }
    color /= (float)ns;

    out[pixel_index] = color;
}

void render(int sx, int sy, int ns, glm::vec3 *out, Camera *cam, BVH<Object> *obj) {
    dim3 threads(8, 8);
    dim3 blocks((sx+threads.x-1) / threads.x, (sy+threads.y-1) / threads.y);
    d_render<<<blocks, threads>>>(sx, sy, ns, out, cam, obj);
}
