#pragma once

#include <stdexcept>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace gamax {
namespace detail {

inline int query_device_id()
{
    int dev = 0;
#if defined(__HIP_PLATFORM_AMD__)
    auto err = hipGetDevice(&dev);
    if (err != hipSuccess) throw std::runtime_error("hipGetDevice failed");
#else
    auto err = cudaGetDevice(&dev);
    if (err != cudaSuccess) throw std::runtime_error("cudaGetDevice failed");
#endif
    return dev;
}

inline int query_sm_count()
{
    int dev = query_device_id();
#if defined(__HIP_PLATFORM_AMD__)
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) throw std::runtime_error("hipGetDeviceProperties failed for SM count");
    return prop.multiProcessorCount;
#else
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) throw std::runtime_error("cudaGetDeviceProperties failed for SM count");
    return prop.multiProcessorCount;
#endif
}

inline int query_max_smem_bytes()
{
    int dev = query_device_id();

#if defined(__HIP_PLATFORM_AMD__)
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) throw std::runtime_error("hipGetDeviceProperties failed for shared memory");
    return prop.sharedMemPerBlock > 0 ? prop.sharedMemPerBlock : 0;
#else
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) throw std::runtime_error("cudaGetDeviceProperties failed for shared memory");
    int smem = 0;
    if (prop.sharedMemPerBlockOptin > 0) smem = prop.sharedMemPerBlockOptin;
    else if (prop.sharedMemPerBlock > 0) smem = prop.sharedMemPerBlock;
    return smem;
#endif
}

} // namespace detail

inline int sm_count()
{
    static int cached = detail::query_sm_count();
    return cached;
}

inline int max_smem_bytes()
{
    static int cached = detail::query_max_smem_bytes();
    return cached;
}

// Warp size is fixed per platform: 32 on CUDA, 64 on HIP. Use a constexpr to keep
// host-side sizing deterministic while hipify remains unaffected.
inline constexpr int warp_size()
{
#if defined(__HIP_PLATFORM_AMD__)
    return 64;
#else
    return 32;
#endif
}

// Legacy names preserved for existing call sites.
inline const int SM_COUNT = sm_count();
inline const int MAX_SMEM_BYTES = max_smem_bytes();

} // namespace gamax
