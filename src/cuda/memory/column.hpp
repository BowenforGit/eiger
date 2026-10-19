// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <cuda_runtime_api.h>
#include <type_traits>

#include "memory/device_buffer.hpp"
#include "memory/scalar.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
namespace gamax {

using column_identity = std::uint64_t;

inline column_identity make_column_identity() noexcept
{
    static std::atomic<column_identity> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

class column_view_base {
public:
    // Borrowed metadata only: constructing/copying a view allocates nothing and
    // enqueues no CUDA work. The owning column/buffer must outlive every
    // asynchronous GPU operation that receives the view.
    column_view_base() = default;

    __host__ __device__ column_view_base(
        void* data,
        size_type size,
        data_type type,
        const size_type offset = 0,
        void* auxiliary = nullptr,
        size_type auxiliary_bytes = 0,
        int32_t auxiliary_tag = 0,
        column_identity identity = 0)
        : _data(data), _size(size - offset), _type(type), _offset(offset)
        , _auxiliary(auxiliary), _auxiliary_bytes(auxiliary_bytes), _auxiliary_tag(auxiliary_tag)
        , _identity(identity) {
            if(offset > size) {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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

    // Non-owning auxiliary metadata used by string views. Keeping these
    // pointers in the trivially-copyable view removes string_column's lazy
    // device allocation and makes concurrent view() calls allocation-free.
    __host__ __device__ void* auxiliary_data() const noexcept { return _auxiliary; }
    __host__ __device__ size_type auxiliary_bytes() const noexcept { return _auxiliary_bytes; }
    __host__ __device__ int32_t auxiliary_tag() const noexcept { return _auxiliary_tag; }
    __host__ __device__ column_identity identity() const noexcept { return _identity; }
    __host__ __device__ size_type offset() const noexcept { return _offset; }

protected:
    void* _data {nullptr};
    size_type _size {0};
    data_type _type;
    size_type _offset {0};
    void* _auxiliary {nullptr};
    size_type _auxiliary_bytes {0};
    int32_t _auxiliary_tag {0};
    column_identity _identity {0};
};

class column_view : public column_view_base {
public:
    column_view() = default;

    __host__ __device__ column_view(
        const void* data, size_type size, data_type type, const size_type offset = 0,
        const void* auxiliary = nullptr, size_type auxiliary_bytes = 0,
        int32_t auxiliary_tag = 0, column_identity identity = 0)
        : column_view_base(
            const_cast<void*>(data), size, type, offset,
            const_cast<void*>(auxiliary), auxiliary_bytes, auxiliary_tag, identity) {}

    template <typename T>
    __host__ __device__ const T* data() const noexcept {
        if constexpr (std::is_void_v<T>) {
            return static_cast<const T*>(_data);
        } else {
            return static_cast<const T*>(_data) + _offset;
        }
    }

    __host__ __device__ const void* data() const noexcept {
        return static_cast<char*>(_data) + _offset * gamax::size_of(_type);
    }
};

class mutable_column_view : public column_view_base {
public:
    mutable_column_view() = default;

    __host__ __device__ mutable_column_view(
        void* data, size_type size, data_type type, const size_type offset = 0,
        void* auxiliary = nullptr, size_type auxiliary_bytes = 0,
        int32_t auxiliary_tag = 0, column_identity identity = 0)
        : column_view_base(
            data, size, type, offset, auxiliary, auxiliary_bytes, auxiliary_tag, identity) {}
    
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

    __host__ __device__ column_view view() const {
        return column_view(
            _data, _size, _type, _offset, _auxiliary, _auxiliary_bytes,
            _auxiliary_tag, _identity);
    }

    void copy_from(column_view const& other, const gamax::size_type n, cudaStream_t stream) {
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
    column() = delete;
    virtual ~column() = default;
    column& operator=(column const& other) = delete;
    column(column const& other)            = delete; // disallow copy constructor to because the column has the exclusive ownership of the data
    column& operator=(column&& other)      = delete;

    column(column&& other) noexcept;
    
    explicit column(
        data_type type,
        size_type num_rows,
        cudaStream_t stream,
        buffer_init init = buffer_init::uninitialized)
        : _size(num_rows)
        , _type(type)
        , _data(num_rows * gamax::size_of(type), stream, init)
        , _identity(make_column_identity()) {}
    
    explicit column(gamax::device_buffer&& buffer, size_type size, data_type type) 
    : _data(std::move(buffer))
    , _size(size)
    , _type(type)
    , _identity(make_column_identity()) {}

    explicit column(data_type dtype,
           size_type size,
           gamax::device_buffer&& data)
        : _type{dtype}
        , _size{size}
        , _data{std::move(data)}
        , _identity(make_column_identity()) {}

    size_type size() const noexcept {
        return _size;
    }

    data_type type() const noexcept {
        return _type;
    }

    device_buffer& buffer() noexcept {
        return _data;
    }

    const device_buffer& buffer() const noexcept {
        return _data;
    }

    void set_size (size_type size) noexcept{
        _size = size;
    }

    gamax::size_type bytes() const noexcept {
        return _data.size();
    }

    column_identity identity() const noexcept { return _identity; }

    virtual std::unique_ptr<column> deep_copy(cudaStream_t stream) const {
        auto clone = std::make_unique<column>(_type, _size, stream);
        cudaMemcpyAsync(clone->data(), data(), _size * gamax::size_of(_type), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        return clone;
    }

    template <typename T = void>
    T* data() const noexcept
    {
        return static_cast<T*>(const_cast<void*>(_data.data()));
    }

    virtual column_view view(const size_type offset = 0) const {
        return column_view(_data.data(), _size, _type, offset, nullptr, 0, 0, _identity);
    }

    virtual mutable_column_view mutable_view(const size_type offset = 0) {
        return mutable_column_view(
            _data.data(), _size, _type, offset, nullptr, 0, 0, _identity);
    }

private:
    gamax::device_buffer _data;
    gamax::size_type _size {0}; // number of elements in the column
    gamax::data_type _type;
    column_identity _identity {0};
};

using column_ptr = std::unique_ptr<column>;

std::unique_ptr<column> make_column(
    data_type type, size_type size, cudaStream_t stream,
    buffer_init init = buffer_init::uninitialized);
std::unique_ptr<column> make_idx_column(
    size_type size, cudaStream_t stream,
    buffer_init init = buffer_init::uninitialized);
void free_column(std::unique_ptr<column> column);

template<typename T>
std::unique_ptr<column> make_column(
    size_type size, cudaStream_t stream,
    buffer_init init = buffer_init::uninitialized) {
    return make_column(data_type{type_to_id<T>()}, size, stream, init);
}

template <typename T>
std::unique_ptr<column> make_column_from_host(
    T* host_data, size_type num_items, cudaStream_t stream) {
    gamax::device_buffer buffer(num_items * sizeof(T), stream);
    cudaMemcpyAsync(buffer.data(), host_data, num_items * sizeof(T), cudaMemcpyHostToDevice, stream);
    return std::make_unique<column>(gamax::data_type{gamax::type_to_id<std::__remove_cvref_t<T>>()}, num_items, std::move(buffer));
}

template <typename T>
std::unique_ptr<column> make_column_from_device(
    T* device_data, size_type num_items, cudaStream_t stream) {
    gamax::device_buffer buffer(num_items * sizeof(T), stream);
    cudaMemcpyAsync(buffer.data(), device_data, num_items * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return std::make_unique<column>(gamax::data_type{gamax::type_to_id<std::__remove_cvref_t<T>>()}, num_items, std::move(buffer));
}

template <typename T>
void copy_column_to_host(column_view col, T* host_data, cudaStream_t stream) {
    cudaMemcpyAsync(host_data, col.data<T>(), col.size() * sizeof(T), cudaMemcpyDeviceToHost, stream);
}

template <typename T>
void copy_column_to_host(column const& col, T* host_data, cudaStream_t stream) {
    copy_column_to_host(col.view(), host_data, stream);
}

template <typename T>
std::unique_ptr<T[]> copy_column_to_host(column_view col, cudaStream_t stream) {
    std::unique_ptr<T[]> host_col(new T[col.size()]);
    copy_column_to_host(col, host_col.get(), stream);
    return host_col;
}

std::unique_ptr<column> make_sequence(
    size_type size,
    cudaStream_t stream,
    gamax::scalar start, 
    gamax::scalar step);

template<typename T = gamax::size_type, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
std::unique_ptr<column> make_sequence(
    size_type size,
    cudaStream_t stream,
    T start = 0, 
    T step = 1)
{
    return make_sequence(size, stream, gamax::scalar(start), gamax::scalar(step));
}

template<typename column_view_type>
std::unique_ptr<column> make_column_from_view(
    column_view_type const& view,
    cudaStream_t stream)
{
    auto col = make_column(view.type(), view.size(), stream);
    cudaMemcpyAsync(col->data(), view.data(), view.size() * gamax::size_of(view.type()), cudaMemcpyDeviceToDevice, stream);
    return col;
}
} // namespace gamax
