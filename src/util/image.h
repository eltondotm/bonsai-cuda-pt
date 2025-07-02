
#pragma once

#include <glm/vec3.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi/stb_image_write.h"

// Correct orientation of buffer to be written as a png
void flip_vertical(int sx, int sy, char *data) {
    for (int j = 0; j < sy/2; ++j) {
        for (int i = 0; i < sx; ++i) {
            int top_idx = j*sx + i;
            int bot_idx = (sy-j-1)*sx + i;
            std::swap(data[top_idx*3],   data[bot_idx*3]);
            std::swap(data[top_idx*3+1], data[bot_idx*3+1]);
            std::swap(data[top_idx*3+2], data[bot_idx*3+2]);
        }
    }
}

// Writes to a png file of specified size with 8-bit rgb color channels
bool write_png(const char *filename, int sx, int sy, char *data) {
    int nchannels = 3;
    flip_vertical(sx, sy, data);
    return stbi_write_png(filename, sx, sy, nchannels, data, sx*nchannels*sizeof(char));
}

// Converts float representation to bytes in order to write to an image file
char *vec_to_byte(glm::vec3 *in, int sx, int sy) {
    char *out = new char[3*sx*sy];
    for (int j = 0; j < sy; ++j) {
        for (int i = 0; i < sx; ++i) {
            int idx_in = j*sx + i;
            int idx_out = 3*idx_in;

            out[idx_out]   = char(in[idx_in].x * 255.99f);
            out[idx_out+1] = char(in[idx_in].y * 255.99f);
            out[idx_out+2] = char(in[idx_in].z * 255.99f);
        }
    }
    return out;
}
