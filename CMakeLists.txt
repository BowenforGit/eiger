cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(GPUOPS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_ROCM "Use ROCm instead of CUDA. [Default=OFF]" OFF)
if(USE_ROCM)
    message(STATUS "Using ROCm.")
    enable_language(HIP)
    set(SubDirName "hip")
    set(LangGPU "HIP")
else()
    message(STATUS "Using CUDA.")
    enable_language(CUDA)
    set(SubDirName "cuda")
    set(LangGPU "CUDA")
endif()

add_library(common INTERFACE)

if(USE_ROCM)
    target_compile_features(common INTERFACE cxx_std_17)
else()
    target_compile_features(common INTERFACE cxx_std_17 cuda_std_17)
endif()

find_package(OpenMP REQUIRED)
MESSAGE(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
MESSAGE(STATUS "OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

if(USE_ROCM)
    if("${OpenMP_CXX_INCLUDE_DIRS}" STREQUAL "")
        list(GET OpenMP_CXX_LIBRARIES 0 OpenMP_CXX_INCLUDE_DIRS)
        cmake_path(GET OpenMP_CXX_INCLUDE_DIRS PARENT_PATH OpenMP_CXX_INCLUDE_DIRS)
        cmake_path(APPEND OpenMP_CXX_INCLUDE_DIRS "include")
        message(STATUS "OpenMP_CXX_INCLUDE_DIRS is manually set to ${OpenMP_CXX_INCLUDE_DIRS}")
    endif()
    target_compile_options(
        common
        INTERFACE
        -O3 
        -Wno-unused-result 
        -Wno-non-c-typedef-for-linkage
        -fopenmp=libgomp
        -I${OpenMP_CXX_INCLUDE_DIRS}
)
    target_compile_definitions(
        common
        INTERFACE 
        $<$<COMPILE_LANGUAGE:HIP>:ROCm_${CMAKE_HIP_ARCHITECTURES}>
    )
else()
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
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

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
endif()

target_compile_definitions(
    common 
    INTERFACE
    $<$<CONFIG:Debug>:DEBUG>
)

option(USE_CUDA_MEMALLOC "Use CUDA to manage memory allocation and deallocation. [Default=OFF]" OFF)
if(USE_CUDA_MEMALLOC)
    message(STATUS "Using CUDA to manage memory allocation and deallocation.")
    target_compile_definitions(
        common 
        INTERFACE
        $<$<COMPILE_LANGUAGE:${LangGPU}>:USE_CUDA_MEMALLOC>
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

if(USE_ROCM)
    CPMAddPackage(
        NAME hipco
        GITHUB_REPOSITORY ROCm/hipCollections
        GIT_TAG main
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
            "BUILD_EXAMPLES OFF"
    )
else()
    CPMAddPackage(
        NAME cuco
        GITHUB_REPOSITORY NVIDIA/cuCollections
        GIT_TAG d92aefb3b64d45c945566d01b9dc500b197095eb
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
            "BUILD_EXAMPLES OFF"
    )

    CPMAddPackage(
        NAME NVTX
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.0-c-cpp
        GIT_SHALLOW TRUE)
    target_link_libraries(common INTERFACE nvtx3-cpp)
    
    add_subdirectory(nvbench)
endif()

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