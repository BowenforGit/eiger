#pragma once

#include <cuda.h>
#include <type_traits>

namespace gamax {
namespace atomic {
    template<typename T>
    using atomic_cas_updatable_t = std::conditional_t<sizeof(T) == 2, unsigned short int, 
                                    std::conditional_t<sizeof(T) == 4, int, 
                                    std::conditional_t<sizeof(T) == 8, unsigned long long, T>>>;
    template<typename T>
    __device__ T atomic_cas(T* address, T compare, T val) {
        using U = atomic_cas_updatable_t<T>;
        return atomicCAS(reinterpret_cast<U*>(address), 
                         static_cast<U>(compare), 
                         static_cast<U>(val));
    }
}
}