
#pragma once

#include <iostream>
#include <string>
#include <filesystem>

static int depth = 0;

static const std::string prefixes[6] = {
    "./",
    "../",
    "../../",
    "../../../",
    "../../../../",
    "../../../../../"
};

// Searches for file in parent directories
std::string read_filepath(const char *filename) {
    for (int i = 0; i < 6; ++i) {
        std::string path = prefixes[i] + filename;
        if (std::filesystem::exists(path)) {
            depth = i;
            return path;
        }
    }
    std::cerr << "File " << filename << " could not be found\n";
    return filename;
}

// Assumes we want to write to the directory the last file was loaded from
std::string write_filepath(const char *filename) {
    return prefixes[depth] + filename;
}
