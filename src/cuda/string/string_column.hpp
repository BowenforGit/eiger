// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <vector>
#include <string>
#include <string_view>

#include "memory/column.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "memory/device_buffer.hpp"

namespace gamax {
enum string_impl_id {
    GERMAN_STRING,
    ARROW_STRING,
    UNSUPPORTED_STRING_IMPL
};

constexpr string_impl_id DEFAULT_STRING_IMPL = ARROW_STRING;

template<typename Base>
class string_column_view_base : public Base {
public:
    string_column_view_base() = default;
    ~string_column_view_base() = default;

    __host__ string_column_view_base(
        const device_buffer& data, 
        const device_buffer& meta, 
        size_type size, 
        string_impl_id impl) 
        : Base(const_cast<void*>(data.data()), size, gamax::data_type{gamax::type_id::STRING}, 0)
        , _meta(const_cast<void*>(meta.data()))
        , _impl(impl)
        , _bytes(data.size()) {}

    /**
     * @brief Construct a string_column_view from a column_view of a string column.
     * 
     * For string columns, the column_view contains a helper buffer with two pointers:
     * [0] = data pointer, [1] = meta pointer (offsets)
     * 
     * @param col A column_view from a string column (must be type_id::STRING)
     * @param impl The string implementation ID
     */
    __host__ __device__ string_column_view_base(
        Base const& col,
        string_impl_id impl)
        : Base(col)
        , _meta(col.auxiliary_data())
        , _impl(impl)
        , _bytes(col.auxiliary_bytes())
    {
    }
    
    __host__ __device__ constexpr string_impl_id impl() const noexcept {
        return _impl;
    }

    // return the number of bytes of the data device_buffer
    __host__ __device__ size_type bytes() const noexcept {
        return _bytes;
    }

    __host__ __device__ auto meta() const noexcept -> decltype(std::declval<Base>().data()) {
        return _meta;
    }

private:
    void* _meta {nullptr};
    string_impl_id _impl;
    gamax::size_type _bytes {0}; 
};

using string_column_view = string_column_view_base<column_view>;
using mutable_string_column_view = string_column_view_base<mutable_column_view>;

class string_column : public column {
public:
    ~string_column() = default;

    string_column(
        gamax::size_type data_bytes,
        gamax::size_type meta_bytes, 
        size_type size, // number of strings
        cudaStream_t stream,
        string_impl_id impl)
        : column(device_buffer(data_bytes, stream), size, gamax::data_type{gamax::type_id::STRING})
        , _meta(device_buffer(meta_bytes, stream))
        , _impl(impl) {}

    string_column(
        gamax::device_buffer&& data, 
        gamax::device_buffer&& meta, 
        size_type size, // number of strings
        string_impl_id impl)
        : column(std::move(data), size, gamax::data_type{gamax::type_id::STRING})
        , _meta(std::move(meta))
        , _impl(impl) {}
    
        
    gamax::device_buffer& meta_buffer() noexcept {
        return _meta;
    }

    const gamax::device_buffer& meta_buffer() const noexcept {
        return _meta;
    }
    
    constexpr string_impl_id impl() const noexcept {
        return _impl;
    }
    
    // the "_s" suffix indicates string column view
    string_column_view view_s() const {
        return string_column_view(this->view(), _impl);
    }

    mutable_string_column_view mutable_view_s() {
        return mutable_string_column_view(this->mutable_view(), _impl);
    }

    column_view view(const size_type offset = 0) const override {
        return column_view(
            this->buffer().data(), this->size(), this->type(), offset,
            _meta.data(), this->buffer().size(), static_cast<int32_t>(_impl),
            this->identity());
    }

    mutable_column_view mutable_view(const size_type offset = 0) override {
        return mutable_column_view(
            this->buffer().data(), this->size(), this->type(), offset,
            _meta.data(), this->buffer().size(), static_cast<int32_t>(_impl),
            this->identity());
    }

private:
    gamax::device_buffer _meta;
    string_impl_id _impl {DEFAULT_STRING_IMPL};
};

using string_column_ptr = std::unique_ptr<string_column>;

class string_scalar_view {
public:
    string_scalar_view() = default;
    ~string_scalar_view() = default;

    __host__ __device__ __forceinline__
    string_scalar_view(const void* data, size_type size) 
        : _data(static_cast<const char*>(data)), _size(size) {}

    __host__ __device__ const char* data() const noexcept {
        return _data;
    }

    __host__ __device__ constexpr size_type size() const noexcept {
        return _size;
    }

    __host__ __device__ const char& operator[](size_type idx) const noexcept {
        return _data[idx];
    }

private:
    const char* _data {nullptr};
    size_type _size {0};
};

class string_scalar {
public:
    string_scalar() = default;
    ~string_scalar() = default;

    string_scalar(std::string_view str, cudaStream_t stream)
        : _size(str.size()), _data(str.size(), stream) {
        cudaMemcpyAsync(_data.data(), str.data(), _size, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    string_scalar_view view() const;

    // returns number of characters in the string
    size_type size() const {
        return _size;
    }

private:
    size_type _size {0};
    gamax::device_buffer _data;
};

inline string_scalar_view string_scalar::view() const {
    return string_scalar_view(_data.data(), _size);
}

std::unique_ptr<string_scalar> make_string_scalar(
    std::string_view str, 
    cudaStream_t stream);

string_column_ptr make_string_column(
    const char* flat_data,
    const int32_t* lengths,
    size_type num_strings,
    cudaStream_t stream,
    string_impl_id impl = DEFAULT_STRING_IMPL);

string_column_ptr make_string_column(
    device_buffer flat_data,
    device_buffer offsets,
    size_type num_strings,
    cudaStream_t stream,
    string_impl_id impl = DEFAULT_STRING_IMPL);
} // namespace gamax
