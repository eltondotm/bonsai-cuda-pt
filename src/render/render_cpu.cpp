
#include <cuda_runtime.h>
#include <glm/vec3.hpp>

#include "util/random_declarations.h"
#include "scene.h"

glm::vec3 trace_ray(const Ray& r, const Scene& scene);

void render_cpu(int sx, int sy, int ns, glm::vec3 *out, Scene scene) {
    for (int j = 0; j < sy; ++j) {
        for (int i = 0; i < sx; ++i) {
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
            color = glm::clamp(color, glm::vec3(0.f), glm::vec3(1.f));

            out[pixel_index] = color;
        }
    }
}
