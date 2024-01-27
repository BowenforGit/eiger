cmake_minimum_required(VERSION 3.20.1)

project(
    GPUJoin 
    LANGUAGES CXX CUDA)

add_subdirectory(nvbench)
add_subdirectory(src)