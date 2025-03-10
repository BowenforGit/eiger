#pragma once

#include <memory>
#include <cuda.h>
#include <variant>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax {
class scalar {
public:
    __host__ __device__ scalar() = default;
    __host__ __device__ scalar(int16_t data) : _data(data), _type(gamax::type_id::INT16) {}
    __host__ __device__ scalar(int32_t data) : _data(data), _type(gamax::type_id::INT32) {}
    __host__ __device__ scalar(int64_t data) : _data(data), _type(gamax::type_id::INT64) {}
    
    template<typename T>
    __host__ __device__ T get() const noexcept {
        return std::get<T>(_data);
    }

    __host__ __device__ gamax::data_type type() const noexcept { return gamax::data_type(_type); }

private:
    gamax::data_type _type;
    std::variant<int16_t, int32_t, int64_t> _data;
};
} // namespace gamax