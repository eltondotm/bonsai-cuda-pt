
#include "scene.h"

#include <curand.h>

// check_cuda and check_curand defined in cuda_errors.h
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
#define checkCurandErrors(val) check_curand( (val), #val, __FILE__, __LINE__ )
void check_curand(curandStatus_t result, char const *const func, const char *const file, int const line);

std::string read_filepath(const char *filename);
std::string write_filepath(const char *filename);

// Util functions for converting between miniScene and glm
inline glm::vec3 mini_to_vec3(mini::vec3f mini) { return glm::vec3(mini.x, mini.y, mini.z); }
inline glm::ivec3 mini_to_ivec3(mini::vec3i mini) { return glm::ivec3(mini.r, mini.s, mini.t); }

// Converting mini::DisneyMaterial to basic materials (cutoffs are arbitrary)
Material mini_disney_to_material(mini::DisneyMaterial::SP mini_mat) {
    glm::vec3 albedo = mini_to_vec3(mini_mat->baseColor);
    if (mini_mat->emission != mini::vec3f(0.f))
        return Material(Emissive{mini_to_vec3(mini_mat->emission)});
    else if (mini_mat->metallic > 0.1f)
        return Material(Metallic{albedo, 1.0f-mini_mat->metallic});
    else if (mini_mat->ior == 1.33f || mini_mat->ior == 2.50f)
        return Material(Glass{mini_mat->ior});
    else if (glm::dot(albedo, albedo) < 0.05f)
        return Material(Metallic{glm::vec3(1.f), 0.f});
    return Material(Lambertian{mini_to_vec3(mini_mat->baseColor)});
};

// Loading a .mini file, then converting to a gpu-friendly format for rendering
Scene create_scene(const char *filename) {
    mini::Scene::SP scene = mini::Scene::load(read_filepath(filename));

    // Gathering unique meshes
    std::set<mini::Mesh::SP> meshes;
    for (const mini::Instance::SP inst : scene->instances) {
        for (const mini::Mesh::SP tri_mesh : inst->object->meshes) {
            meshes.insert(tri_mesh);
        }
    }

    Scene converted_scene;
    std::vector<Object *>emitters;

    std::vector<Object> tri_bvhs;
    for (const mini::Mesh::SP& mesh : meshes) {
        // Moving mesh data to arrays for access from host or device
        glm::vec3 *verts;
        glm::vec3 *norms;
        int v_size = mesh->vertices.size()*sizeof(glm::vec3);
        int n_size = mesh->normals.size()*sizeof(glm::vec3);
        checkCudaErrors(cudaMallocManaged((void **)&verts, v_size));
        checkCudaErrors(cudaMallocManaged((void **)&norms, n_size));
        checkCudaErrors(cudaMemcpy(verts, mesh->vertices.data(), v_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(norms, mesh->normals.data(), n_size, cudaMemcpyHostToDevice));

        Material material = mini_disney_to_material(mesh->material->as<mini::DisneyMaterial>());

        TriangleMesh *tri_mesh;
        checkCudaErrors(cudaMallocManaged((void**)&tri_mesh, sizeof(TriangleMesh)));
        *tri_mesh = TriangleMesh{verts, norms, material, !mesh->normals.empty()};

        std::vector<Object> tri_vec;
        for (int i = 0; i < mesh->getNumPrims(); ++i) {
            tri_vec.push_back(Object(Triangle(tri_mesh, mini_to_ivec3(mesh->indices[i]))));
        }

        // BVH construction
        std::vector<BVHNode> bvh_vec = build_bvh(tri_vec, SAH);

        int n_bytes_tri = tri_vec.size()*sizeof(Object);
        int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);

        Object *tri;
        checkCudaErrors(cudaMallocManaged((void **)&tri, n_bytes_tri));
        checkCudaErrors(cudaMemcpy(tri, tri_vec.data(), n_bytes_tri, cudaMemcpyHostToDevice));

        BVHNode *bvh;
        checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
        checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

        tri_bvhs.push_back(BVH<Object>(tri, tri_vec.size(), bvh, bvh_vec.size()));

        if (const Emissive *e = cuda::std::get_if<Emissive>(&material)) {
            for (int i = 0; i < tri_vec.size(); ++i) {
                emitters.push_back(&(tri[i]));
            }
        }
    }

    std::vector<BVHNode> scn_bvh_vec = build_bvh(tri_bvhs, SAH);

    int n_bytes_scn = tri_bvhs.size()*sizeof(Object);
    int n_bytes_bvh = scn_bvh_vec.size()*sizeof(BVHNode);
    int n_bytes_emt = emitters.size()*sizeof(Object *);

    Object *scn;
    checkCudaErrors(cudaMallocManaged((void **)&scn, n_bytes_scn));
    checkCudaErrors(cudaMemcpy(scn, tri_bvhs.data(), n_bytes_scn, cudaMemcpyHostToDevice));

    BVHNode *scn_bvh;
    checkCudaErrors(cudaMallocManaged((void **)&scn_bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(scn_bvh, scn_bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    BVH<Object> *obj;
    checkCudaErrors(cudaMallocManaged((void **)&obj, sizeof(BVH<Object>)));
    *obj = BVH<Object>(scn, tri_bvhs.size(), scn_bvh, scn_bvh_vec.size());
    
    Object **emt;
    checkCudaErrors(cudaMallocManaged((void **)&emt, n_bytes_emt));
    checkCudaErrors(cudaMemcpy(emt, emitters.data(), n_bytes_emt, cudaMemcpyHostToDevice));

    converted_scene.geometry = obj;
    converted_scene.emitters = emt;
    converted_scene.n_emitters = emitters.size();
    // Camera will be initialized in main

    return converted_scene;
}

// Deallocates a BVH of triangle mesh BVHs
void free_tri_mesh_bvh(const BVH<Object> *geo, int idx) {
    const BVHNode& node = geo->bvh[idx];
    const Object *prims = geo->prims;

    if (node.num_hitables != 0) {
        const BVH<Object>& tri_bvh = cuda::std::get<BVH<Object>>(prims[node.start_idx].underlying);
        const Triangle& tri = cuda::std::get<Triangle>(tri_bvh.prims[0].underlying);

        checkCudaErrors(cudaFree(tri.mesh->vertices));
        checkCudaErrors(cudaFree(tri.mesh->normals));
        checkCudaErrors(cudaFree(tri.mesh));
        checkCudaErrors(cudaFree(tri_bvh.prims));
        checkCudaErrors(cudaFree(tri_bvh.bvh));
        return;
    }

    free_tri_mesh_bvh(geo, idx + 1);
    free_tri_mesh_bvh(geo, node.one_idx);
}

// Assumes two-level BVH (which is true for any scene built with create_scene())
void free_scene(const Scene& scene) {
    free_tri_mesh_bvh(scene.geometry, 0);

    checkCudaErrors(cudaFree(scene.geometry->prims));
    checkCudaErrors(cudaFree(scene.geometry->bvh));
    checkCudaErrors(cudaFree(scene.geometry));
    checkCudaErrors(cudaFree(scene.camera));
    checkCudaErrors(cudaFree(scene.emitters));
}
