cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(GPUOPS LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_CUDA "Use CUDA. [Default=ON]" OFF)
option(USE_ROCM "Use ROCm. [Default=OFF]" OFF)
option(BUILD_TEST "Build tests. [Default=ON]" ON)
option(BUILD_BENCHMARK "Build benchmarks. [Default=ON]" ON)
option(USE_DEFAULT_MEMALLOC "Use the default memory allocation and deallocation. [Default=OFF]" OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(USE_CUDA ON)
    MESSAGE(STATUS "CUDA is enabled.")
    enable_language(CUDA)
    set(SubDirName "cuda")
    set(LangGPU "CUDA")
else()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        set(USE_ROCM ON)
        MESSAGE(STATUS "ROCm is enabled.")
        enable_language(HIP)
        set(SubDirName "hip")
        set(LangGPU "HIP")
    else()
        MESSAGE(FATAL_ERROR "Neither CUDA nor ROCm is enabled.")
    endif()
endif()

add_library(common INTERFACE)

if(USE_ROCM)
    target_compile_features(common INTERFACE cxx_std_17 hip_std_17)
else(USE_CUDA)
    target_compile_features(common INTERFACE cxx_std_17 cuda_std_17)
endif()

# manually set the include path for ROCm platform
# so that find_package(OpenMP) can find the OpenMP headers
if(USE_ROCM)
    set(OpenMP_CXX_INCLUDE_DIR "/usr/lib/gcc/x86_64-linux-gnu/11/include")
endif()

find_package(OpenMP REQUIRED)
MESSAGE(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
MESSAGE(STATUS "OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

if(USE_ROCM)
    set(HIP_COMPILER_FLAGS "")
    list(APPEND HIP_COMPILER_FLAGS -O3)
    list(APPEND HIP_COMPILER_FLAGS -Wno-unused-result -Wno-non-c-typedef-for-linkage)
    target_include_directories(
        common
        INTERFACE
        ${OpenMP_CXX_INCLUDE_DIRS}
    )
    target_compile_options(
        common
        INTERFACE
        $<$<COMPILE_LANGUAGE:HIP>:${HIP_COMPILER_FLAGS}>
    )
    target_compile_definitions(
        common
        INTERFACE 
        $<$<COMPILE_LANGUAGE:HIP>:ROCm_${CMAKE_HIP_ARCHITECTURES}>
    )
else(USE_CUDA)
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
endif()

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

if(USE_DEFAULT_MEMALLOC)
    message(STATUS "Using CUDA to manage memory allocation and deallocation.")
    target_compile_definitions(
        common 
        INTERFACE
        $<$<COMPILE_LANGUAGE:${LangGPU}>:USE_DEFAULT_MEMALLOC>
    )
endif()

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# hipify the code
if(USE_ROCM)
    execute_process(
        COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/build_amd.py
        RESULT_VARIABLE hipify_return_value
    )
    if (NOT hipify_return_value EQUAL 0)
        message(FATAL_ERROR "Failed to hipify files!")
    endif()
endif()

add_subdirectory(src/${SubDirName})