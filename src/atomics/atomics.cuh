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

    template<typename T>
    __device__ __forceinline__ T atomic_add(T* address, T val) {
        return atomicAdd(address, val);
    }

    template<>
    __device__ __forceinline__ int64_t atomic_add<int64_t>(int64_t* address, int64_t val) {
        return atomicAdd(reinterpret_cast<unsigned long long*>(address), 
                         static_cast<unsigned long long>(val));
    }

    template<>
    __device__ __forceinline__ long long int atomic_add<long long int>(long long int* address, long long int val) {
        return atomicAdd(reinterpret_cast<unsigned long long*>(address), 
                         static_cast<unsigned long long>(val));
    }

    template<>
    __device__ __forceinline__ int16_t atomic_add<int16_t>(int16_t* address, int16_t val) {
        GAMAX_UNREACHABLE("atomicAdd does not support 16-bit integers");
    }

    template<typename T>
    __device__ __forceinline__ T atomic_max(T* address, T val) {
        return atomicMax(address, val);
    }

    template<>
    __device__ __forceinline__ int64_t atomic_max<int64_t>(int64_t* address, int64_t val) {
        return atomicMax(reinterpret_cast<long long int*>(address), 
                         static_cast<long long int>(val));
    }

    template<>
    __device__ __forceinline__ int16_t atomic_max<int16_t>(int16_t* address, int16_t val) {
        GAMAX_UNREACHABLE("atomicMax does not support 16-bit integers");
    }
}
}