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

constexpr string_impl_id DEFAULT_STRING_IMPL = GERMAN_STRING;

template<typename Base>
class string_column_view_base : public Base {
public:
    string_column_view_base() = default;
    ~string_column_view_base() = default;

    __host__ __device__ string_column_view_base(
        const device_buffer& data, 
        const device_buffer& meta, 
        size_type size, 
        string_impl_id impl) 
        : Base(data.data(), size, gamax::data_type{gamax::type_id::STRING}, 0)
        , _meta(meta.data())
        , _impl(impl)
        , _bytes(data.size()) {}

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
    // Default constructor for empty string column
    string_column() : column(gamax::data_type{gamax::type_id::STRING}, 0) {};
    ~string_column() = default;

    string_column(
        gamax::size_type data_bytes,
        gamax::size_type meta_bytes, 
        size_type size, // number of strings
        string_impl_id impl)
        : column(device_buffer(data_bytes, false), size, gamax::data_type{gamax::type_id::STRING})
        , _meta(device_buffer(meta_bytes, false))
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
        return string_column_view(buffer(), _meta, this->size(), _impl);
    }

    mutable_string_column_view mutable_view_s() {
        return mutable_string_column_view(buffer(), _meta, this->size(), _impl);
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

    string_scalar(std::string_view str, cudaStream_t stream = cudaStreamDefault) 
        : _size(str.size()), _data(str.size(), false, stream) {
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
    cudaStream_t stream = cudaStreamDefault);

string_column_ptr make_string_column(
    const char* flat_data,
    const int32_t* lengths,
    size_type num_strings,
    cudaStream_t stream = cudaStreamDefault,
    string_impl_id impl = DEFAULT_STRING_IMPL);

string_column_ptr make_string_column(
    device_buffer flat_data,
    device_buffer offsets,
    size_type num_strings,
    cudaStream_t stream = cudaStreamDefault,
    string_impl_id impl = DEFAULT_STRING_IMPL);
} // namespace gamax