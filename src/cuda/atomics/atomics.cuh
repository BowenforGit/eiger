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
        // sm_60+ has a native atomicAdd(double*, double). Without this
        // branch the fall-through path below truncates the double via
        // static_cast<unsigned long long> and writes integer bytes into
        // a double cell — SUM(FLOAT64) groupby results then read back
        // as subnormal garbage.
        if constexpr (std::is_same_v<T, double>) {
            return atomicAdd(address, val);
        } else {
            return atomicAdd(reinterpret_cast<unsigned long long*>(address),
                             static_cast<unsigned long long>(val));
        }
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

    template<>
    __device__ __forceinline__ double atomic_max<double>(double* addr, double value) {
        if(!signbit(value)) {
            return __longlong_as_double(atomicMax((long long int*)addr, __double_as_longlong(value)));
        } else {
            double v_ptr[1];
            v_ptr[0] = value;

            unsigned long long int temp[1];
            temp[0] = atomicMin((unsigned long long int*)addr, reinterpret_cast<unsigned long long int*>(v_ptr)[0]);
            
            return reinterpret_cast<double*>(temp)[0];
        }
    }

    template<>
    __device__ __forceinline__ float atomic_max<float>(float* addr, float value) {
        float old;
        old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
            __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

        return old;
    }

    template<typename T, std::enable_if_t<sizeof(T) < 4, int> = 0>
    __device__ __forceinline__ T atomic_min(T* address, T val) {
        GAMAX_UNREACHABLE("atomicMin does not support types smaller than 4 bytes");
    }

    template<typename T, std::enable_if_t<sizeof(T) >= 4 && sizeof(T) < 8, int> = 0>
    __device__ __forceinline__ T atomic_min(T* address, T val) {
        return atomicMin(address, val);
    }

    template<typename T, std::enable_if_t<sizeof(T) == 8, int> = 0>
    __device__ __forceinline__ T atomic_min(T* address, T val) {
        return atomicMin(reinterpret_cast<long long int*>(address), 
                         static_cast<long long int>(val));
    }

    template<>
    __device__ __forceinline__ double atomic_min<double>(double* addr, double value) {
        if(!signbit(value)) {
            return __longlong_as_double(atomicMin((long long int*)addr, __double_as_longlong(value)));
        } else {
            double v_ptr[1];
            v_ptr[0] = value;

            unsigned long long int temp[1];
            temp[0] = atomicMax((unsigned long long int*)addr, reinterpret_cast<unsigned long long int*>(v_ptr)[0]);
            
            return reinterpret_cast<double*>(temp)[0];
        }
    }
    

    template<>
    __device__ __forceinline__ float atomic_min<float>(float* addr, float value) {
        float old;
        old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
            __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

        return old;
    }
} // namespace atomic
} // namespace gamax