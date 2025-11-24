#pragma once

#include <cuda.h>
#include <type_traits>

#include "utils/assert.cuh"

namespace gamax {
namespace atomic {
    template<typename T>
    using atomic_cas_updatable_t = std::conditional_t<sizeof(T) == 2, unsigned short int, 
                                    std::conditional_t<sizeof(T) == 4, int, 
                                    std::conditional_t<sizeof(T) == 8, unsigned long long, T>>>;
    template<typename T>
    __device__ __forceinline__ T atomic_cas(T* address, T compare, T val) {
        using U = atomic_cas_updatable_t<T>;
        return atomicCAS(reinterpret_cast<U*>(address), 
                         static_cast<U>(compare), 
                         static_cast<U>(val));
    }
    
    template<typename T, std::enable_if_t<sizeof(T) < 4, int> = 0>
    __device__ __forceinline__ T atomic_add(T* address, T val) {
        GAMAX_UNREACHABLE("[Eiger] atomicAdd does not support types smaller than 4 bytes");
    }

    template<typename T, std::enable_if_t<sizeof(T) >= 4 && sizeof(T) < 8, int> = 0>
    __device__ __forceinline__ T atomic_add(T* address, T val) {
        return atomicAdd(address, val);
    }

    template<typename T, std::enable_if_t<sizeof(T) == 8, int> = 0>
    __device__ __forceinline__ T atomic_add(T* address, T val) {
        return atomicAdd(reinterpret_cast<unsigned long long*>(address), 
                         static_cast<unsigned long long>(val));
    }

    template<typename T, std::enable_if_t<sizeof(T) < 4, int> = 0>
    __device__ __forceinline__ T atomic_max(T* address, T val) {
        GAMAX_UNREACHABLE("atomicMax does not support types smaller than 4 bytes");
    }

    template<typename T, std::enable_if_t<sizeof(T) >= 4 && sizeof(T) < 8, int> = 0>
    __device__ __forceinline__ T atomic_max(T* address, T val) {
        return atomicMax(address, val);
    }

    template<typename T, std::enable_if_t<sizeof(T) == 8, int> = 0>
    __device__ __forceinline__ T atomic_max(T* address, T val) {
        return atomicMax(reinterpret_cast<long long int*>(address), 
                         static_cast<long long int>(val));
    }
} // namespace atomic
} // namespace gamax