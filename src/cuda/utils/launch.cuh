#pragma once

#include <iostream>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define GAMAX_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE hipOccupancyMaxPotentialBlockSize
#define GAMAX_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MP hipOccupancyMaxActiveBlocksPerMultiprocessor
#define GAMAX_FUNC_SET_ATTRIBUTE hipFuncSetAttribute
#else
#include <cuda.h>
#include <cuda_runtime.h>
#define GAMAX_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE cudaOccupancyMaxPotentialBlockSize
#define GAMAX_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MP cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define GAMAX_FUNC_SET_ATTRIBUTE cudaFuncSetAttribute
#endif

#include "utils/device_info.hpp"
#include "utils/check_error.hpp"

inline void print_launch_params(const char* const func, int grid_size, int block_size, int dynamic_smem_size, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Launching " << func 
              << " with grid size " << grid_size 
              << ", block size " << block_size 
              << ", dynamic shared memory size " << dynamic_smem_size 
              << ", and stream " << stream << std::endl;
#endif
}

#define LAUNCH_MAX_OCCUPANCY(func, dynamicSMemSize, stream, ...) \
    { \
        int minGridSize = 0, blockSize = 0; \
        CHECK_CUDA_ERROR(GAMAX_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE(&minGridSize, &blockSize, func, dynamicSMemSize, 0)); \
        print_launch_params(#func, minGridSize, blockSize, dynamicSMemSize, stream); \
        func<<<minGridSize, blockSize, dynamicSMemSize, stream>>>(__VA_ARGS__); \
        CHECK_LAST_CUDA_ERROR(); \
    }
