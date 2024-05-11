cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#   set(CMAKE_CUDA_ARCHITECTURES 86)
# endif()

project(
    GPUOPS 
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA_MEMALLOC "Use CUDA to manage memory allocation and deallocation. [Default=OFF]" OFF)

add_library(common INTERFACE)
target_compile_features(common INTERFACE cxx_std_17 cuda_std_17)

set(NVCC_COMPILER_FLAGS "")
list(APPEND NVCC_COMPILER_FLAGS -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES})
list(APPEND NVCC_COMPILER_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --extended-lambda)
list(APPEND NVCC_COMPILER_FLAGS -Xptxas="-v" -lineinfo)
list(APPEND NVCC_COMPILER_FLAGS -Xcompiler -fopenmp -O3 --compile)

target_compile_options(
    common
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_COMPILER_FLAGS}>
)

target_include_directories(
    common 
    INTERFACE 
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/>/"
)

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

target_compile_definitions(
    common
    INTERFACE 
    $<$<COMPILE_LANGUAGE:CUDA>:SM${CMAKE_CUDA_ARCHITECTURES}0>
)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
endif()

target_compile_definitions(
    common 
    INTERFACE
    $<$<CONFIG:Debug>:DEBUG>
)

if(USE_CUDA_MEMALLOC)
    message(STATUS "Using CUDA to manage memory allocation and deallocation.")
    target_compile_definitions(
        common 
        INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:USE_CUDA_MEMALLOC>
    )
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
find_package(GTest)

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

add_subdirectory(nvbench)
add_subdirectory(src)