#pragma once
#include "memory/column.hpp"
#include "memory/scalar.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"


namespace gamax::expression::detail {
struct column_access_helper {
    template <typename T>
    __host__ __device__ gamax::scalar operator()(gamax::column_view col, gamax::size_type i) const noexcept {
        return gamax::scalar(col.data<T>()[i]);
    }

    template <typename T>
    __host__ __device__ T operator()(const T* data, gamax::size_type i) const noexcept {
        return data[i];
    }
};

struct negate_helper {
    template <typename T>
    __host__ __device__ gamax::scalar operator()(gamax::scalar value) const {
        return gamax::scalar(-value.get<T>());
    }

    template <typename T>
    __host__ __device__ T operator()(T value) const {
        return -value;
    }
};

struct left_shift_helper {
    template <typename T>
    __host__ __device__ gamax::scalar operator()(gamax::scalar value, int bits) const {
        return gamax::scalar(this->operator()(value.get<T>(), bits));
    }

    template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    __host__ __device__ T operator()(T value, int bits) const {
        return value << bits;
    }

    template <typename T, typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
    __host__ __device__ T operator()(T value, int bits) const {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
            throw std::runtime_error("Only integral types can be left-shifted.");
#else
            GAMAX_UNREACHABLE("Only integral types can be left-shifted.");
#endif
    }
};

struct right_shift_helper {
    template <typename T>
    __host__ __device__ gamax::scalar operator()(gamax::scalar value, int bits) const {
        return gamax::scalar(this->operator()(value.get<T>(), bits));
    }

    template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    __host__ __device__ T operator()(T value, int bits) const {
        return value >> bits;
    }

    template <typename T, typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
    __host__ __device__ T operator()(T value, int bits) const {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
            throw std::runtime_error("Only integral types can be right-shifted.");
#else
            GAMAX_UNREACHABLE("Only integral types can be right-shifted.");
#endif
    }
};

struct add_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(e1.get<T>() + e2.get<U>());
    }

    template <typename T>
    __host__ __device__ T operator()(T lhs, T rhs) const {
        return lhs + rhs;
    }
};

struct substract_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(e1.get<T>() - e2.get<U>());
    }

    template <typename T>
    __host__ __device__ T operator()(T lhs, T rhs) const {
        return lhs - rhs;
    }
};

struct multiply_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(e1.get<T>() * e2.get<U>());
    }

    template <typename T>
    __host__ __device__ T operator()(T lhs, T rhs) const {
        return lhs * rhs;
    }
};

struct divide_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(e1.get<T>() / e2.get<U>());
    }

    template <typename T>
    __host__ __device__ T operator()(T lhs, T rhs) const {
        return lhs / rhs;
    }
};

struct gt_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(static_cast<bool>(e1.get<T>() > e2.get<U>()));
    }

    template <typename T>
    __host__ __device__ bool operator()(T lhs, T rhs) const {
        return lhs > rhs;
    }
};

struct lt_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(static_cast<bool>(e1.get<T>() < e2.get<U>()));
    }

    template <typename T>
    __host__ __device__ bool operator()(T lhs, T rhs) const {
        return lhs < rhs;
    }
};

struct ge_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(static_cast<bool>(e1.get<T>() >= e2.get<U>()));
    }

    template <typename T>
    __host__ __device__ bool operator()(T lhs, T rhs) const {
        return lhs >= rhs;
    }
};

struct le_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(static_cast<bool>(e1.get<T>() <= e2.get<U>()));
    }

    template <typename T>
    __host__ __device__ bool operator()(T lhs, T rhs) const {
        return lhs <= rhs;
    }
};

struct eq_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(static_cast<bool>(e1.get<T>() == e2.get<U>()));
    }

    template <typename T>
    __host__ __device__ bool operator()(T lhs, T rhs) const {
        return lhs == rhs;
    }
};

struct neq_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(static_cast<bool>(e1.get<T>() != e2.get<U>()));
    }

    template <typename T>
    __host__ __device__ bool operator()(T lhs, T rhs) const {
        return lhs != rhs;
    }
};

struct and_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(static_cast<bool>(e1.get<T>() && e2.get<U>()));
    }

    template <typename T>
    __host__ __device__ bool operator()(T lhs, T rhs) const {
        return lhs && rhs;
    }
};

struct or_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(static_cast<bool>(e1.get<T>() || e2.get<U>()));
    }

    template <typename T>
    __host__ __device__ bool operator()(T lhs, T rhs) const {
        return lhs || rhs;
    }
};

struct select_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2, gamax::scalar e3) const {
        return gamax::scalar(e1.get<bool>() ? e2.get<T>() : e3.get<U>());
    }

    template <typename T>
    __host__ __device__ T operator()(bool cond, T true_value, T false_value) const {
        return cond ? true_value : false_value;
    }
};
} // namespace gamax::expression::detail