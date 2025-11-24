#pragma once

#include <memory>
#include <cuda_runtime_api.h>
#include <variant>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax {
class scalar {
public:
    using scalar_data_type = union {
        bool bool8_v;
        int8_t int8_v;
        int16_t int16_v;
        int32_t int32_v;
        int64_t int64_v;
    };
    __host__ __device__ scalar() = default;
    __host__ __device__ scalar(type_id type, scalar_data_type data) 
        : _data(data), _type(type) {}
    __host__ __device__ constexpr scalar(bool data) 
        : _data{.bool8_v = static_cast<gamax::id_to_type_impl<gamax::type_id::BOOL8>::type>(data)}
        , _type(gamax::type_id::BOOL8) {}
    __host__ __device__ constexpr scalar(int8_t data) : _data{.int8_v = data}, _type(gamax::type_id::INT8) {}
    __host__ __device__ constexpr scalar(int16_t data) : _data{.int16_v = data}, _type(gamax::type_id::INT16) {}
    __host__ __device__ constexpr scalar(int32_t data) : _data{.int32_v = data}, _type(gamax::type_id::INT32) {}
    __host__ __device__ constexpr scalar(int64_t data) : _data{.int64_v = data}, _type(gamax::type_id::INT64) {}
    
    template<typename T>
    __host__ __device__ constexpr T get() const noexcept {
        if constexpr (std::is_same_v<T, bool>) {
            return _data.bool8_v;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return _data.int8_v;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return _data.int16_v;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return _data.int32_v;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return _data.int64_v;
        } else {
            // Fallback or error handling if needed, though static_assert is better
            return T{};
        }
    }

    template<typename T>
    __host__ __device__ constexpr void set(T value) noexcept {
        if constexpr (std::is_same_v<T, bool>) {
            _data.bool8_v = value;
            _type = gamax::type_id::BOOL8;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            _data.int8_v = value;
            _type = gamax::type_id::INT8;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            _data.int16_v = value;
            _type = gamax::type_id::INT16;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            _data.int32_v = value;
            _type = gamax::type_id::INT32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            _data.int64_v = value;
            _type = gamax::type_id::INT64;
        }
    }

    __host__ __device__ constexpr gamax::data_type type() const noexcept { return gamax::data_type(_type); }

    gamax::type_id _type;
    scalar_data_type _data;
};
} // namespace gamax