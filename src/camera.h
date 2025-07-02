
#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ray.h"

class Camera {
public:
    __host__ __device__ Camera() :
        looking_at(0.f, 0.f, 0.f),
        rotation(glm::vec3(0.f)),
        radius(5.f),
        vfov(90.f),
        aspect(1.7778f),
        aperture(0.f),
        focal_dist(5.f) {
        view_update();
    }

    __host__ __device__ Ray generate_ray(glm::vec2 uv) {
        float h_view = 2.f * tan(glm::radians(vfov) * 0.5f);
        float w_view = h_view * aspect;
        glm::vec2 view_dim = glm::vec2(w_view, h_view);
        glm::vec2 view_dir = (uv - 0.5f) * view_dim;

        Ray r = Ray(glm::vec3(0.f), glm::vec3(view_dir, -1.0f));
        r.transform(iview);
        return r;
    }

    __host__ __device__ void view_update() {
        glm::vec3 orientation = glm::mat3_cast(rotation) * glm::vec3(0.f, 0.f, 1.f);
        position = looking_at + radius * orientation;
        iview = glm::translate(glm::mat4(1.f), position) * glm::mat4_cast(rotation);
        view = glm::inverse(iview);
    }

    __host__ __device__ void look_at(glm::vec3 pos, glm::vec3 target) {
        position = pos;
        looking_at = target;
        radius = glm::length(pos - target);

        glm::vec3 dir = glm::normalize(pos - target);
        glm::vec3 up  = glm::vec3(0.f, 1.f, 1.f);
        rotation = glm::quatLookAt(dir, up);
        view_update();
    }

    __host__ __device__ void set_rotation_euler(glm::vec3 euler) { rotation = glm::quat(glm::radians(euler)); view_update(); }
    __host__ __device__ void set_rotation_dir(glm::vec3 dir) { rotation = glm::quatLookAt(dir, glm::vec3(0.f, 1.f, 0.f)); view_update(); }
    __host__ __device__ void set_radius(float r) { radius = r; view_update(); }
    __host__ __device__ void set_fov(float fov) { vfov = fov; }
    __host__ __device__ void set_aspect(float asp) { aspect = asp; }
    __host__ __device__ void set_aperture(float ap) { aperture = ap; }
    __host__ __device__ void set_focal_dist(float fd) { focal_dist = fd; }

private:
    glm::vec3 position;
    glm::vec3 looking_at;
    glm::quat rotation;
    float radius;

    float vfov;
    float aspect;

    // Todo: focusing
    float aperture;
    float focal_dist;

    glm::mat4 view;
    glm::mat4 iview;
};
