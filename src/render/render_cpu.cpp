
#include <cuda_runtime.h>
#include <glm/vec3.hpp>

#include "camera.h"
#include "object.h"

glm::vec3 trace_ray(const Ray& r, Object *obj);

void render_cpu(int sx, int sy, glm::vec3 *out, Camera *cam, Object *obj) {
    for (int j = 0; j < sy; ++j) {
        for (int i = 0; i < sx; ++i) {
            int pixel_index = j*sx + i;
            float u = (float)i/sx;
            float v = (float)j/sy;

            glm::vec2 uv((float)i/sx, (float)j/sy);
            Ray r = cam->generate_ray(uv);

            out[pixel_index] = trace_ray(r, obj);
        }
    }
}
