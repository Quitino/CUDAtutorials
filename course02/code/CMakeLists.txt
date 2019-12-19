cmake_minimum_required(VERSION 2.8)
project(square)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(square square.cu) 


 

