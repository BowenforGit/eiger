#pragma once

#include <limits>
#include <cstdint>
#include <type_traits>

#include "types/data_type.hpp"

namespace gamax {
    template<typename T>
    __host__ __device__ constexpr T sentinel() {
        if constexpr (std::is_signed_v<T>) {
            return std::numeric_limits<T>::min();
        } else {
            return std::numeric_limits<T>::max();
        }
    }
}