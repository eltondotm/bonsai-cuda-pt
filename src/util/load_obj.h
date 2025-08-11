
/*
 * obj_load.h
 * Simple function for loading no-material obj files.
 * Line prefixes must be one of v, vn, vt, f.
 * Triangulates n-gons.
 * Adapted from miniScene (see ext/miniScene)
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <cstring>
#include <fstream>

#include <glm/vec3.hpp>

// TODO: add texcoord support
struct ObjMesh {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::ivec3> faces;

  void clear() {
    vertices.clear();
    normals.clear();
    faces.clear();
  }
};

std::vector<ObjMesh> load_obj(const std::string &obj_filename) {
  std::vector<ObjMesh> meshes;
  ObjMesh mesh;

  std::map<int, int> knownVertices;
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;

  std::string line;
  std::ifstream in(obj_filename.c_str());
  while (in) {
    std::getline(in,line);
    if (!in.good()) break;
    if (line[0] == 'v' && line[1] == ' ') {
      glm::vec3 v;
      sscanf((char*)line.c_str(),"v %f %f %f",&v.x,&v.y,&v.z);
      vertices.push_back(v);
    } else if (line[0] == 'v' && line[1] == 'n') {
      glm::vec3 n;
      sscanf((char*)line.c_str(), "vn %f %f %f",&n.x,&n.y,&n.z);
      normals.push_back(n);
    } else if (line[0] == 'v' && line[1] == 't') {
      continue;  // Texture support not implemented yet
    } else if (line[0] == '#' && line[1] == ' ') {
      continue;
    } else if (line[0] == 'g' && line[1] == ' ') {
      meshes.push_back(mesh);
      mesh.clear();
      knownVertices.clear();
    } else if (line[0] == 'f' && line[1] == ' ') {
      char *s = strtok((char*)line.c_str()," \n\t");
      std::vector<int> face;
      while (true) {
        s = strtok(nullptr," \n\t");
        if (!s) break;
        face.push_back(std::stoi(s));
      }
      for (auto &i : face) {
        if (knownVertices.find(i) == knownVertices.end()) {
          knownVertices[i] = mesh.vertices.size();
          mesh.vertices.push_back(vertices[i-1]);
          mesh.normals.push_back(normals[i-1]);
        } 
        i = knownVertices[i];
      }
      for (int i=2;i<face.size();i++)
        mesh.faces.push_back({face[0],face[i-1],face[i]});
    } else
      throw std::runtime_error("not simple obj input format: "+line);
  }
  meshes.push_back(mesh);
  return meshes;
}
