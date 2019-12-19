cmake_minimum_required(VERSION 2.8)
project(course04)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(histo histo.cu)
CUDA_ADD_EXECUTABLE(histo1 histo1.cu) 


