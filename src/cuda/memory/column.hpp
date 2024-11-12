#pragma once

#include <memory>
#include <cuda.h>

#include "memory/device_buffer.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "utils/sequence.cuh"

namespace gamax {

class column_view_base {
public:
    column_view_base() = default;

    __host__ __device__ column_view_base(void* data, size_type size, data_type type, const size_type offset = 0) 
        : _data(data), _size(size - offset), _type(type), _offset(offset) {
            if(offset > size) {
#ifndef __CUDA_ARCH__
                throw std::runtime_error("Offset is greater than the size of the column");
#else
                gamax_assert("Offset is greater than the size of the column");
#endif
            }
        }

    __host__ __device__ size_type size() const noexcept {
        return _size;
    }

    __host__ __device__ data_type type() const noexcept {
        return _type;
    }

protected:
    void* _data {nullptr};
    size_type _size {0};
    data_type _type;
    size_type _offset {0};
};

class column_view : public column_view_base {
public:
    column_view() = default;

    __host__ __device__ column_view(void* data, size_type size, data_type type, const size_type offset = 0) 
        : column_view_base(data, size, type, offset) {}

    template <typename T>
    __host__ __device__ const T* data() const noexcept {
        return static_cast<const T*>(_data) + _offset;
    }

    __host__ __device__ const void* data() const noexcept {
        return static_cast<char*>(_data) + _offset * gamax::size_of(_type);
    }
};

class mutable_column_view : public column_view_base {
public:
    mutable_column_view() = default;

    __host__ __device__ mutable_column_view(void* data, size_type size, data_type type, const size_type offset = 0) 
        : column_view_base(data, size, type, offset) {}
    
    void set_size (size_type size) noexcept{
        _size = size;
    }

    template <typename T>
    __host__ __device__ T* data() const noexcept {
        return static_cast<T*>(_data) + _offset;
    }

    __host__ __device__ void* data() const noexcept {
        return static_cast<char*>(_data) + _offset * gamax::size_of(_type);
    }

    void copy_from(column_view const& other, const gamax::size_type n, cudaStream_t stream = cudaStreamDefault) {
        if(_type != other.type()) {
            throw std::runtime_error("Cannot copy between columns of different types");
        }
        if(_size < n || other.size() < n) {
            throw std::runtime_error("Column is too small to copy from other");
        }
        
        cudaMemcpyAsync(this->data(), other.data(), n * gamax::size_of(_type), cudaMemcpyDeviceToDevice, stream);
    }
};

class column {
public:
    column() = default;
    ~column() = default;
    column& operator=(column const& other) = delete;
    column(column const& other)            = delete; // disallow copy constructor to because the column has the exclusive ownership of the data
    column& operator=(column&& other)      = delete;

    column(column&& other) noexcept;
    
    explicit column(data_type type, size_type num_rows, bool clear = false) 
        : _size(num_rows)
        , _type(type)
        , _data(num_rows * gamax::size_of(type), clear) {}
    
    __host__ __device__ explicit column(gamax::device_buffer&& buffer, size_type size, data_type type) 
    : _data(std::move(buffer))
    , _size(size)
    , _type(type) {}

    explicit column(data_type dtype,
           size_type size,
           gamax::device_buffer&& data)
        : _type{dtype}
        , _size{size}
        , _data{std::move(data)} {}

    __host__ __device__ size_type size() const noexcept {
        return _size;
    }

    __host__ __device__ data_type type() const noexcept {
        return _type;
    }

    __host__ __device__ device_buffer& buffer() noexcept {
        return _data;
    }

    void set_size (size_type size) noexcept{
        _size = size;
    }

    std::unique_ptr<column> deep_copy(cudaStream_t stream = cudaStreamDefault) const {
        auto clone = std::make_unique<column>(_type, _size);
        cudaMemcpyAsync(clone->data(), data(), _size * gamax::size_of(_type), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        return clone;
    }

    template <typename T = void>
    __host__ __device__ T* data() const noexcept
    {
        return static_cast<T*>(_data.data());
    }

    __host__ __device__ column_view view(const size_type offset = 0) const {
        return column_view(_data.data(), _size, _type, offset);
    }

    __host__ __device__ mutable_column_view mutable_view(const size_type offset = 0) {
        return mutable_column_view(_data.data(), _size, _type, offset);
    }

private:
    gamax::device_buffer _data;
    gamax::size_type _size {0}; // number of elements in the column
    gamax::data_type _type;
};

std::unique_ptr<column> make_column(data_type type, size_type size, bool clear=false); 
std::unique_ptr<column> make_idx_column(size_type size, bool clear=false); 
void free_column(std::unique_ptr<column> column);

template<typename T>
std::unique_ptr<column> make_column(size_type size, bool clear=false) {
    return make_column(data_type{type_to_id<T>()}, size, clear);
}

template <typename T>
std::unique_ptr<column> make_column_from_host(T* host_data, size_type num_items) {
    gamax::device_buffer buffer(num_items * sizeof(T), false);
    cudaMemcpy(buffer.data(), host_data, num_items * sizeof(T), cudaMemcpyHostToDevice);
    return std::make_unique<column>(gamax::data_type{gamax::type_to_id<T>()}, num_items, std::move(buffer));
}

template <typename T>
std::unique_ptr<column> make_column_from_device(T* device_data, size_type num_items) {
    gamax::device_buffer buffer(num_items * sizeof(T), false);
    cudaMemcpy(buffer.data(), device_data, num_items * sizeof(T), cudaMemcpyDeviceToDevice);
    return std::make_unique<column>(gamax::data_type{gamax::type_to_id<T>()}, num_items, std::move(buffer));
}

template <typename T>
void copy_column_to_host(column_view col, T* host_data) {
    cudaMemcpy(host_data, col.data<T>(), col.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
void copy_column_to_host(column const& col, T* host_data) {
    // cudaMemcpy(host_data, col.data<T>(), col.size() * sizeof(T), cudaMemcpyDeviceToHost);
    copy_column_to_host(col.view(), host_data);
}

template <typename T>
std::unique_ptr<T[]> copy_column_to_host(column_view col) {
    std::unique_ptr<T[]> host_col(new T[col.size()]);
    copy_column_to_host(col, host_col.get());
    return host_col;
}

template<typename T = gamax::size_type>
std::unique_ptr<column> make_sequence(
    size_type size, 
    T start = 0, 
    T step = 1) {
    
    auto col = make_column(data_type{type_to_id<T>()}, size);
    gamax::utils::device::create_sequence(col->data<T>(), size, start, step);
    return col;
}

template<typename column_view_type>
std::unique_ptr<column> make_column_from_view(
    column_view_type const& view, 
    cudaStream_t stream = cudaStreamDefault) 
{
    auto col = make_column(view.type(), view.size());
    cudaMemcpyAsync(col->data(), view.data(), view.size() * gamax::size_of(view.type()), cudaMemcpyDeviceToDevice, stream);
    return col;
}
} // namespace gamax