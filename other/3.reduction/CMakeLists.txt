cmake_minimum_required(VERSION 2.8)
project(reduction)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(1B-reduction 1B-reduction.cu) 
CUDA_ADD_EXECUTABLE(reduction_1 reduction_1.cu) 
CUDA_ADD_EXECUTABLE(reduction_2 reduction_2.cu) 

 

