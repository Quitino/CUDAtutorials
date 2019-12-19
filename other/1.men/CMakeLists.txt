cmake_minimum_required(VERSION 2.8)
project(mem)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(mem mem.cu) 
CUDA_ADD_EXECUTABLE(mem-host mem-host.cu) 

 

