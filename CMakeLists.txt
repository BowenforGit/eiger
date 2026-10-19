# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# Bootstrap one RAPIDS 26.06 dependency table before any dependency is
# configured.  RMM 26.06 is built against this exact cohort (notably CCCL and
# cuCollections), so allowing a nested project to initialize RAPIDS first can
# produce an unsupported mixture of versions.
set(rapids-cmake-version "26.06")
set(rapids-cmake-tag "v26.06.00")
set(_gamax_rapids_bootstrap "${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake")
file(
  DOWNLOAD
  https://raw.githubusercontent.com/rapidsai/rapids-cmake/v26.06.00/RAPIDS.cmake
  "${_gamax_rapids_bootstrap}"
  EXPECTED_HASH SHA256=f21479e8ba5af680d9302a8d1f5cba4e0505072a1d8b4c5482a15ae7ecdaefd3
  TLS_VERIFY ON
)
include("${_gamax_rapids_bootstrap}")
include(rapids-cpm)
rapids_cpm_init()

project(GPUOPS LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TEST "Build tests. [Default=ON]" ON)
option(BUILD_BENCHMARK "Build benchmarks. [Default=ON]" ON)
option(BUILD_TPCH "Build TPC-H query runner (requires Arrow and Parquet). [Default=OFF]" OFF)
option(BUILD_ALPS "Build ALPS multi-GPU component (requires NCCL + MPI). [Default=OFF]" OFF)

# ALPS ships a FindNCCL module and registers ctest tests from
# src/cuda/alps (a sibling of src/cuda/tests), so its module path must be
# discoverable and testing must be enabled at the top level for ctest to
# find them.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(BUILD_TEST OR BUILD_ALPS)
  enable_testing()
endif()
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(SubDirName "cuda")
message(STATUS "CUDA is enabled (the ROCm/hipify build path is no longer supported).")

add_library(common INTERFACE)

target_compile_features(common INTERFACE cxx_std_17 cuda_std_17)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
MESSAGE(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
MESSAGE(STATUS "OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # no need to set the -gencode flag for because cmake will automatically add it based on CMAKE_CUDA_ARCHITECTURES
    # no need to add -fopenmp because it is automatically added by cmake when linking to OpenMP::OpenMP_CXX
    set(NVCC_COMPILER_FLAGS "")
    list(APPEND NVCC_COMPILER_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --extended-lambda)
    list(APPEND NVCC_COMPILER_FLAGS -lineinfo)
    list(APPEND NVCC_COMPILER_FLAGS --compile)

    # Add optimization flags based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND NVCC_COMPILER_FLAGS -Xcompiler -O3)
    else()
        list(APPEND NVCC_COMPILER_FLAGS -g -G)
    endif()

    target_compile_options(
        common
        INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_COMPILER_FLAGS}>
    )
    target_compile_definitions(
        common
        INTERFACE 
        $<$<COMPILE_LANGUAGE:CUDA>:SM${CMAKE_CUDA_ARCHITECTURES}0>
    )

target_include_directories(
    common 
    INTERFACE 
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/${SubDirName}/>/"
)

target_compile_definitions(
    common 
    INTERFACE
    $<$<CONFIG:Debug>:DEBUG>
)

# Use a shared CPM source cache so dependencies are not re-downloaded across build directories.
set(CPM_SOURCE_CACHE "$ENV{HOME}/.cache/CPM" CACHE PATH "CPM shared source cache")

add_subdirectory(src/${SubDirName})
