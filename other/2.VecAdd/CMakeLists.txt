cmake_minimum_required(VERSION 2.8)
project(VecAdd)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(VecAdd vec_add.cu)  

