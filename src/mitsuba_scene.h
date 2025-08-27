
#pragma once

#include <vector>
#include <map>
#include <string>

#include <tinyparser-mitsuba.h>

#include "util/file.h"
#include "util/load_obj.h"
#include "util/texture.h"
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
    // return glm::mat4(
    //     m[0],  m[1],  m[2],  m[3],
    //     m[4],  m[5],  m[6],  m[7],
    //     m[8],  m[9],  m[10], m[11],
    //     m[12], m[13], m[14], m[15]
    // );
}

// Creates a single-node BVH to hold the primitive as well as transform/material information
// This format makes instancing convenient, as multiple BVHs can point to the same object
BVH<Object> construct_primitive(Object& prim, Transform *transforms) {
    std::vector<Object> prm_vec{prim};
    std::vector<BVHNode> bvh_vec = build_bvh(prm_vec, SAH, transforms);

    int n_bytes_prm = prm_vec.size()*sizeof(Object);
    int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);

    Object *prm;
    checkCudaErrors(cudaMallocManaged((void **)&prm, n_bytes_prm));
    checkCudaErrors(cudaMemcpy(prm, prm_vec.data(), n_bytes_prm, cudaMemcpyHostToDevice));

    BVHNode *bvh;
    checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    return BVH<Object>(prm, prm_vec.size(), bvh, bvh_vec.size());
}

// Creates a BVH for a triangle mesh
BVH<Object> construct_mesh(const std::string& path, uint16_t mat, uint16_t trans, Transform *transforms) {
    std::vector<ObjMesh> meshes = load_obj(path);

    ObjMesh& mesh = meshes[0];

    glm::vec3 *verts;
    glm::vec3 *norms;
    float     *u;
    float     *v;
    int v_size  = mesh.vertices.size()*sizeof(glm::vec3);
    int n_size  = mesh.normals.size()*sizeof(glm::vec3);
    int us_size = mesh.us.size()*sizeof(float);
    int vs_size = mesh.vs.size()*sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&verts, v_size));
    checkCudaErrors(cudaMallocManaged((void **)&norms, n_size));
    checkCudaErrors(cudaMallocManaged((void **)&u,    us_size));
    checkCudaErrors(cudaMallocManaged((void **)&v,    vs_size));
    checkCudaErrors(cudaMemcpy(verts, mesh.vertices.data(), v_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(norms, mesh.normals.data(), n_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(u, mesh.us.data(), us_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(v, mesh.vs.data(), vs_size, cudaMemcpyHostToDevice));

    TriangleMesh *tri_mesh;
    checkCudaErrors(cudaMallocManaged((void**)&tri_mesh, sizeof(TriangleMesh)));
    *tri_mesh = TriangleMesh{verts, norms, u, v, !mesh.normals.empty()};

    std::vector<Object> tri_vec;
    for (int i = 0; i < mesh.faces.size(); ++i) {
        Triangle tri = Triangle(tri_mesh, mesh.faces[i]);
        tri_vec.push_back(Object(std::move(tri), mat, trans));
    }

    // Getting information for cached BVH
    std::string bvh_path(path);
    bvh_path.replace(bvh_path.find("models"), 6, "bvhs");
    bvh_path.replace(bvh_path.end() - 3, bvh_path.end(), "bvh");

    // BVH construction / loading
    BVHNode *bvh;
    int nnodes;
    // TODO: Must support object sorting to load BVH data
    if (false) {
        std::ifstream source(bvh_path, std::ios::in | std::ios::binary);
        if (!source.is_open()) {
            std::cout << "Error opening BVH file.\n";
        }
        std::vector<unsigned char> bvh_data((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
        source.close();
        nnodes = bvh_data.size() / sizeof(BVHNode);

        checkCudaErrors(cudaMallocManaged((void **)&bvh, bvh_data.size()));
        checkCudaErrors(cudaMemcpy(bvh, bvh_data.data(), bvh_data.size(), cudaMemcpyHostToDevice));
    } else {
        std::vector<BVHNode> bvh_vec = build_bvh(tri_vec, SAH, transforms);
        int n_bytes_bvh = bvh_vec.size()*sizeof(BVHNode);
        nnodes = bvh_vec.size();
        std::string new_dir = bvh_path.substr(0, bvh_path.find_last_of("/"));
        std::filesystem::create_directory(new_dir);
        std::ofstream dest(bvh_path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!dest.is_open()) {
            std::cout << "Error creating BVH file \n";
        }
        dest.write(reinterpret_cast<const char*>(bvh_vec.data()), bvh_vec.size()*sizeof(BVHNode));
        dest.close();

        checkCudaErrors(cudaMallocManaged((void **)&bvh, n_bytes_bvh));
        checkCudaErrors(cudaMemcpy(bvh, bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));
    }
    
    Object *tri;
    int n_bytes_tri = tri_vec.size()*sizeof(Object);
    checkCudaErrors(cudaMallocManaged((void **)&tri, n_bytes_tri));
    checkCudaErrors(cudaMemcpy(tri, tri_vec.data(), n_bytes_tri, cudaMemcpyHostToDevice));

    return BVH<Object>(tri, tri_vec.size(), bvh, nnodes);
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
Camera parse_camera(const tpm::Object *cam, int& width, int& height) {
    const auto& props = cam->properties();
    float fov = props.at("fov").getNumber();
    glm::mat4 to_world = transform_to_mat4(props.at("to_world").getTransform());

    for (const std::shared_ptr<tpm::Object>& child : cam->anonymousChildren()) {
        if (child->type() == tpm::OT_FILM) {
            auto& film_props = child->properties();
            width = film_props.at("width").getInteger();
            height = film_props.at("height").getInteger();
        }
    }

    Camera camera;
    camera.set_fov(fov);
    camera.set_view(to_world);
    return camera;
}

// Converts Mitsuba materials to materials implemented in this renderer
// Maps material name to their index in the vector
void parse_material(const tpm::Object *m, std::vector<Material>& materials, std::map<std::string, uint16_t>& id_to_idx,
                    std::map<std::string, cudaTextureObject_t>& textures, const std::string& path) {
    while(!m->hasID()) {
        m = m->anonymousChildren()[0].get();
    }          

    // Maps material id to its location in the materials vector
    id_to_idx[m->id()] = materials.size();

    // Ignore material add-ons like twosided, fetch base material
    while (m->anonymousChildren().size() != 0) {
        m = m->anonymousChildren()[0].get();
    }

    // Note: tpm::Propertie get() methods take an optional bool for error checking
    // It would be worth adding these checks if some fields return default values
    const std::string& type = m->pluginType();
    const auto& props = m->properties();
    if (type == "roughdielectric" || type == "smoothdielectric" || type == "thindielectric") {
        float ior = props.at("int_ior").getNumber();
        Material mat = Glass{ior};
        materials.push_back(mat);
    } else if (type == "conductor") {
        Material mat = Metallic{};
        materials.push_back(mat);
    } else {
        // Default to diffuse
        glm::vec3 albedo(1.f);
        cudaTextureObject_t tex_obj = -1ULL;
        if (props.find("reflectance") != props.end())
            albedo = color_to_vec3(props.at("reflectance").getColor());
        else if (props.find("diffuse_reflectance") != props.end())
            albedo = color_to_vec3(props.at("diffuse_reflectance").getColor());
        else if (props.find("specular_reflectance") != props.end())
            albedo = color_to_vec3(props.at("specular_reflectance").getColor());
        
        // Textured material
        const auto& nc = m->namedChildren();
        if (!nc.empty()) {
            std::shared_ptr<tpm::Object> tex = nullptr;
            if (nc.find("reflectance") != nc.end())
                tex = nc.at("reflectance");
            else if (nc.find("diffuse_reflectance") != nc.end())
                tex = nc.at("diffuse_reflectance");
            else if (nc.find("specular_reflectance") != nc.end())
                tex = nc.at("specular_reflectance");
            if (tex) {
                std::string tex_path = path + "/" + tex->properties().at("filename").getString();
                auto tex_loc = textures.find(tex_path);
                if (tex_loc == textures.end()) {
                    tex_obj = load_texture(tex_path);
                    textures[tex_path] = tex_obj;
                } else {
                    tex_obj = tex_loc->second;
                }
            }
        } 
        Material mat = Lambertian{albedo, tex_obj};
        materials.push_back(mat);
    }
}

BVH<Object> parse_shape(const tpm::Object *o, 
                        std::string& path, 
                        std::vector<Material>& materials, 
                        std::map<std::string, uint16_t>& id_to_idx,
                        std::vector<Transform>& transforms,
                        std::map<Transform, uint16_t>& t_to_idx) {
    // Getting transformation matrix
    const auto& props = o->properties();
    const tpm::Transform& tpm_to_world = props.at("to_world").getTransform();
    glm::mat4 to_world = transform_to_mat4(tpm_to_world);

    // Retrieving index or adding to vector if not present
    uint16_t t_idx;
    auto t_it = t_to_idx.find(to_world);
    if (t_it == t_to_idx.end()) {
        t_idx = transforms.size();
        t_to_idx[to_world] = t_idx;
        transforms.push_back(to_world);
    } else {
        t_idx = t_it->second;
    }

    // Getting material (no pointers for now)
    unsigned int mat_idx = 0;
    const auto& children = o->anonymousChildren();
    for (const std::shared_ptr<tpm::Object>& child : children) {
        if (child->type() == tpm::OT_BSDF) {
            mat_idx = id_to_idx[child->id()];
        }
        else if (child->type() == tpm::OT_EMITTER) {
            const tpm::Color& radiance = child->properties().at("radiance").getColor();
            Material e = Emissive{color_to_vec3(radiance)};
            mat_idx = materials.size();
            materials.push_back(e);
            break;
        }
    }

    // Constructing object
    const std::string& type = o->pluginType();
    if (type == "obj") {
        return construct_mesh(path + "/" + props.at("filename").getString(), mat_idx, t_idx, transforms.data());
    } else if (type == "rectangle") {
        //return construct_square(mat_idx, t_idx, transforms.data());
        Object prim = Object(Square(), mat_idx, t_idx);
        return construct_primitive(prim, transforms.data());
    } else if (type == "cube") {
        return construct_cube(mat_idx, t_idx, transforms.data());
    }
    return construct_cube(mat_idx, t_idx, transforms.data());
}

// Loading a Mitsuba .xml file, then converting to a gpu-friendly format for rendering
Scene create_mitsuba_scene(const char *filename, int& width, int& height) {
    std::string path = read_filepath(filename);
    tpm::SceneLoader loader = tpm::SceneLoader();
    const tpm::Scene scene = loader.loadFromFile(path + "/scene_v3.xml");
    const std::vector<std::shared_ptr<tpm::Object>>& objects = scene.anonymousChildren();

    std::vector<Material> materials;
    std::map<std::string, uint16_t> id_to_idx;

    std::vector<Transform> transforms{ Transform() };
    std::map<Transform, uint16_t> t_to_idx;
    t_to_idx[glm::mat4(1.f)] = 0;

    std::map<std::string, cudaTextureObject_t> textures;

    std::vector<Object> geometry;

    Scene converted_scene;

    for (const std::shared_ptr<tpm::Object>& object : objects) {
        tpm::ObjectType type = object->type();
        switch (type) {
            case tpm::OT_SENSOR:
                Camera *cam;
                checkCudaErrors(cudaMallocManaged((void **)&cam, sizeof(Camera)));
                *cam = parse_camera(object.get(), width, height);
                converted_scene.camera = cam;
                break;
            case tpm::OT_BSDF:
                parse_material(object.get(), materials, id_to_idx, textures, path);
                break;
            case tpm::OT_SHAPE:
                BVH<Object> shape = parse_shape(object.get(), path, materials, id_to_idx, transforms, t_to_idx);
                geometry.push_back(Object(std::move(shape)));
                break;
        }
    }

    std::vector<BVHNode> scn_bvh_vec = build_bvh(geometry, SAH, transforms.data());

    int n_bytes_scn = geometry.size()*sizeof(Object);
    int n_bytes_bvh = scn_bvh_vec.size()*sizeof(BVHNode);
    int n_bytes_mat = materials.size()*sizeof(Material);
    int n_bytes_trn = transforms.size()*sizeof(Transform);

    Object *scn;
    checkCudaErrors(cudaMallocManaged((void **)&scn, n_bytes_scn));
    checkCudaErrors(cudaMemcpy(scn, geometry.data(), n_bytes_scn, cudaMemcpyHostToDevice));

    BVHNode *scn_bvh;
    checkCudaErrors(cudaMallocManaged((void **)&scn_bvh, n_bytes_bvh));
    checkCudaErrors(cudaMemcpy(scn_bvh, scn_bvh_vec.data(), n_bytes_bvh, cudaMemcpyHostToDevice));

    BVH<Object> *obj;
    checkCudaErrors(cudaMallocManaged((void **)&obj, sizeof(BVH<Object>)));
    *obj = BVH<Object>(scn, geometry.size(), scn_bvh, scn_bvh_vec.size());

    Material *mat;
    checkCudaErrors(cudaMallocManaged((void **)&mat, n_bytes_mat));
    checkCudaErrors(cudaMemcpy(mat, materials.data(), n_bytes_mat, cudaMemcpyHostToDevice));

    Transform *trn;
    checkCudaErrors(cudaMallocManaged((void **)&trn, n_bytes_trn));
    checkCudaErrors(cudaMemcpy(trn, transforms.data(), n_bytes_trn, cudaMemcpyHostToDevice));
    
    converted_scene.geometry = obj;
    converted_scene.materials = mat;
    converted_scene.transforms = trn;
    converted_scene.n_emitters = 0;

    return converted_scene;
}
