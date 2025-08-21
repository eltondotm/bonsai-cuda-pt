
#pragma once

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stbi/stb_image.h"

#include "cuda_errors.h"

using uchar = unsigned char;

cudaTextureObject_t load_texture(const std::string& path) {
    // Loading image file
    int x,y,n;
    uchar *data = stbi_load(path.c_str(), &x, &y, &n, 4);

    // Allocating underlying data array
    cudaArray_t texArray;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&texArray, &channelDesc, x, y));

    // Pitch is the width in bytes of the array (including padding)
    const size_t spitch = x * sizeof(uchar) * 4;
    checkCudaErrors(cudaMemcpy2DToArray(texArray, 0, 0, data, spitch,
                                        x * sizeof(uchar) * 4, y, cudaMemcpyHostToDevice));

    stbi_image_free(data);

    // Set texture resource information and parameters
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = texArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = true;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.readMode = cudaReadModeNormalizedFloat;

    // Creating texture object
    cudaTextureObject_t tex = 0;
    checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDesc, NULL));

    return tex;
}
