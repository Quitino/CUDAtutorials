cmake_minimum_required(VERSION 2.8)
project(course03)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(atomics atomics.cu) 
CUDA_ADD_EXECUTABLE(homework_scan homework_scan.cu) 
CUDA_ADD_EXECUTABLE(memory memory.cu) 
CUDA_ADD_EXECUTABLE(reduce reduce.cu) 
CUDA_ADD_EXECUTABLE(scan scan.cu) 


 

