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
        uint8_t uint8_v;
        uint16_t uint16_v;
        uint32_t uint32_v;
        uint64_t uint64_v;
        float float32_v;
        double float64_v;
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
    __host__ __device__ constexpr scalar(uint8_t data) : _data{.uint8_v = data}, _type(gamax::type_id::UINT8) {}
    __host__ __device__ constexpr scalar(uint16_t data) : _data{.uint16_v = data}, _type(gamax::type_id::UINT16) {}
    __host__ __device__ constexpr scalar(uint32_t data) : _data{.uint32_v = data}, _type(gamax::type_id::UINT32) {}
    __host__ __device__ constexpr scalar(uint64_t data) : _data{.uint64_v = data}, _type(gamax::type_id::UINT64) {}
    __host__ __device__ constexpr scalar(float data) : _data{.float32_v = data}, _type(gamax::type_id::FLOAT32) {}
    __host__ __device__ constexpr scalar(double data) : _data{.float64_v = data}, _type(gamax::type_id::FLOAT64) {}
    
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
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return _data.uint8_v;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return _data.uint16_v;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return _data.uint32_v;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            return _data.uint64_v;
        } else if constexpr (std::is_same_v<T, float>) {
            return _data.float32_v;
        } else if constexpr (std::is_same_v<T, double>) {
            return _data.float64_v;
        } else {
            static_assert(sizeof(T) == 0, "Unsupported type for scalar::get");
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
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            _data.uint8_v = value;
            _type = gamax::type_id::UINT8;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            _data.uint16_v = value;
            _type = gamax::type_id::UINT16;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            _data.uint32_v = value;
            _type = gamax::type_id::UINT32;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            _data.uint64_v = value;
            _type = gamax::type_id::UINT64;
        } else if constexpr (std::is_same_v<T, float>) {
            _data.float32_v = value;
            _type = gamax::type_id::FLOAT32;
        } else if constexpr (std::is_same_v<T, double>) {
            _data.float64_v = value;
            _type = gamax::type_id::FLOAT64;
        } else {
            static_assert(sizeof(T) == 0, "Unsupported type for scalar::set");
        }
    }

    __host__ __device__ constexpr gamax::data_type type() const noexcept { return gamax::data_type(_type); }

    gamax::type_id _type;
    scalar_data_type _data;
};
} // namespace gamax