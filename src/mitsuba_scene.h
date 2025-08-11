
#pragma once

#include <vector>
#include <map>
#include <string>

#include <tinyparser-mitsuba.h>

#include "util/file.h"
#include "util/load_obj.h"
#include "scene.h"
#include "cube.h"

namespace tpm = tinyparser_mitsuba;


// Converts tpm::Color to glm::vec3
inline glm::vec3 color_to_vec3(const tpm::Color& c) {
    return glm::vec3(c.r, c.g, c.b);
}

// Converts tpm::Transform to glm::mat4
inline glm::mat4 transform_to_mat4(const tpm::Transform& t) {
    // tpm::Transform stored in row-major format
    auto& m = t.matrix;
    return glm::mat4(
        m[0],  m[4],  m[8],  m[12],
        m[1],  m[5],  m[9],  m[13],
        m[2],  m[6],  m[10], m[14],
        m[3],  m[7],  m[11], m[15]
    );
}

// Creates a single-node BVH to hold the primitive as well as transform/material information
// This format makes instancing convenient, as multiple BVHs can point to the same object
BVH<Object> construct_primitive(Object& prim, const Material& mat, const glm::mat4& trans) {
    std::vector<Object> prm_vec{prim};
    std::vector<BVHNode> bvh_vec = build_bvh(prm_vec, SAH);

    int n_bytes_prm = prm_vec.size()*sizeof(Object);
    int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);

    Object *prm;
    checkCudaErrors(cudaMallocManaged((void **)&prm, n_bytes_prm));
    checkCudaErrors(cudaMemcpy(prm, prm_vec.data(), n_bytes_prm, cudaMemcpyHostToDevice));

    BVHNode *bvh;
    checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    return BVH<Object>(prm, prm_vec.size(), bvh, bvh_vec.size(), mat, trans);
}

// Creates a BVH for a triangle mesh
BVH<Object> construct_mesh(std::string& path, const Material& mat, const glm::mat4& trans) {
    std::vector<ObjMesh> meshes = load_obj(path);

    ObjMesh& mesh = meshes[0];

    glm::vec3 *verts;
    glm::vec3 *norms;
    int v_size = mesh.vertices.size()*sizeof(glm::vec3);
    int n_size = mesh.normals.size()*sizeof(glm::vec3);
    checkCudaErrors(cudaMallocManaged((void **)&verts, v_size));
    checkCudaErrors(cudaMallocManaged((void **)&norms, n_size));
    checkCudaErrors(cudaMemcpy(verts, mesh.vertices.data(), v_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(norms, mesh.normals.data(), n_size, cudaMemcpyHostToDevice));

    TriangleMesh *tri_mesh;
    checkCudaErrors(cudaMallocManaged((void**)&tri_mesh, sizeof(TriangleMesh)));
    *tri_mesh = TriangleMesh{verts, norms, !mesh.normals.empty()};

    std::vector<Object> tri_vec;
    for (int i = 0; i < mesh.faces.size(); ++i) {
        Triangle tri = Triangle(tri_mesh, mesh.faces[i]);
        tri_vec.push_back(Object(std::move(tri)));
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

    return BVH<Object>(tri, tri_vec.size(), bvh, bvh_vec.size(), mat, trans);
}

/* tinyparser-mitsuba scene format
 * -------------------------------
 * Tree structure; every node is represented as an object with a type, ID, properties, and children.
 * Types:
 * OT_SCENE
 *   - All scene objects are stored in mChildren
 * OT_INTEGRATOR
 *   - Ignored in this case, since this renderer has only one integrator
 * OT_SENSOR
 *   - Camera transform and parameters in mProperties (keys "to_world" and "fov")
 * OT_BSDF
 *   - mPluginType specifies the material type (or the modifier, such as two-sided or blended)
 *   - mID gives a unique identifier that objects can reference, only set on the outer layer
 *   - Parameters in mProperties, see Mitsuba documentation for a per-material list
 * OT_SHAPE
 *   - Specifies either a primitive or a mesh, given by mPluginType
 *   - Transform information in mProperties ("to_world")
 *   - Material pointer in mChildren
 */

// Initialize camera parameters from sensor object
Camera parse_camera(const tpm::Object *cam) {
    const auto& props = cam->properties();
    float fov = props.at("fov").getNumber();
    glm::mat4 to_world = transform_to_mat4(props.at("to_world").getTransform());

    Camera camera;
    camera.set_fov(fov);
    camera.set_view(to_world);
    return camera;
}

// Returns the index of the newly added material
void parse_material(const tpm::Object *m, std::vector<Material>& materials, std::map<std::string, int>& id_to_idx) {
    // Maps material id to its location in the materials vector
    id_to_idx[m->id()] = materials.size();

    Material mat = Lambertian{glm::vec3(1.f)};
    materials.push_back(mat);
    return;

    // Ignore material add-ons like twosided, fetch base material
    while (m->anonymousChildren().size() != 0) {
        m = m->anonymousChildren()[0].get();
    }

    // Note: tpm::Propertie get() methods take an optional bool for error checking
    // It would be worth adding these checks if some fields return default values
    const std::string& type = m->pluginType();
    const auto& props = m->properties();
    if (type == "diffuse") {
        const tpm::Color& albedo = props.at("reflectance").getColor();
        Material mat = Lambertian{color_to_vec3(albedo)};
        materials.push_back(mat);
    } else {
        // Default to white diffuse
        Material mat = Lambertian{glm::vec3(1.f)};
        materials.push_back(mat);
    }
}

BVH<Object> parse_shape(const tpm::Object *o, std::string& path, std::vector<Material>& materials, std::map<std::string, int>& id_to_idx) {
    // Getting transformation matrix
    const auto& props = o->properties();
    const tpm::Transform& to_world = props.at("to_world").getTransform();
    glm::mat4 trans = transform_to_mat4(to_world);

    // Getting material (no pointers for now)
    Material mat;
    const auto& children = o->anonymousChildren();
    for (const std::shared_ptr<tpm::Object>& child : children) {
        if (child->type() == tpm::OT_BSDF) {
            int mat_idx = id_to_idx[child->id()];
            mat = materials[mat_idx];
        }
        else if (child->type() == tpm::OT_EMITTER) {
            const tpm::Color& radiance = child->properties().at("radiance").getColor();
            mat = Emissive{color_to_vec3(radiance)};
            break;
        }
    }

    //trans = glm::mat4(1.f);

    // Constructing object
    const std::string& type = o->pluginType();
    if (type == "obj") {
        return construct_mesh(path + "/" + props.at("filename").getString(), mat, trans);
    } else if (type == "rectangle") {
        Object prim = Object(Square());
        return construct_primitive(prim, mat, trans);
    } else if (type == "cube") {
        return construct_cube(mat, trans);
    }
    return construct_cube(mat, trans);
}

// Loading a .mini file, then converting to a gpu-friendly format for rendering
Scene create_mitsuba_scene(const char *filename) {
    std::string path = read_filepath(filename);
    tpm::SceneLoader loader = tpm::SceneLoader();
    const tpm::Scene scene = loader.loadFromFile(path + "/scene_v3.xml");
    const std::vector<std::shared_ptr<tpm::Object>>& objects = scene.anonymousChildren();

    std::vector<Material> materials;
    std::map<std::string, int> id_to_idx;

    std::vector<Object> geometry;

    Scene converted_scene;

    for (const std::shared_ptr<tpm::Object>& object : objects) {
        tpm::ObjectType type = object->type();
        switch (type) {
            case tpm::OT_SENSOR:
                Camera *cam;
                checkCudaErrors(cudaMallocManaged((void **)&cam, sizeof(Camera)));
                *cam = parse_camera(object.get());
                converted_scene.camera = cam;
            case tpm::OT_BSDF:
                parse_material(object.get(), materials, id_to_idx);
                break;
            case tpm::OT_SHAPE:
                BVH<Object> shape = parse_shape(object.get(), path, materials, id_to_idx);
                geometry.push_back(Object(std::move(shape)));
                break;
        }
    }

    std::vector<BVHNode> scn_bvh_vec = build_bvh(geometry, SAH);

    int n_bytes_scn = geometry.size()*sizeof(Object);
    int n_bytes_bvh = scn_bvh_vec.size()*sizeof(BVHNode);

    Object *scn;
    checkCudaErrors(cudaMallocManaged((void **)&scn, n_bytes_scn));
    checkCudaErrors(cudaMemcpy(scn, geometry.data(), n_bytes_scn, cudaMemcpyHostToDevice));

    BVHNode *scn_bvh;
    checkCudaErrors(cudaMallocManaged((void **)&scn_bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(scn_bvh, scn_bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    BVH<Object> *obj;
    checkCudaErrors(cudaMallocManaged((void **)&obj, sizeof(BVH<Object>)));
    *obj = BVH<Object>(scn, geometry.size(), scn_bvh, scn_bvh_vec.size());
    
    converted_scene.geometry = obj;
    converted_scene.n_emitters = 0;

    return converted_scene;
}
