cmake_minimum_required (VERSION 3.17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project ("bonsai-cuda-pt" LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(MAIN "src/main.cpp")
set(RENDER "src/render/render_cpu.cpp" "src/render/render_kernel.cu")
set(OBJECTS "src/objects/bvh_build.cpp")

add_executable (pt ${MAIN} ${RENDER} ${OBJECTS})

set_target_properties(pt PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)

target_compile_options(pt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g --expt-relaxed-constexpr>)

target_include_directories(pt PRIVATE 
  "src" 
  "ext" 
  "src/render"
  "src/objects" 
  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_link_libraries(pt PRIVATE CUDA::cudart_static)
