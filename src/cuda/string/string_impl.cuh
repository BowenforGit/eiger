#pragma once

#include <cuda.h>
#include <type_traits>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "memory/column.hpp"
#include "memory/device_buffer.hpp"
#include "utils/assert.cuh"
#include "string/string_column.hpp"

namespace gamax::detail {

// Policy trait validation
template<typename T, typename = void>
struct is_valid_header_policy : std::false_type {};

template<typename T>
struct is_valid_header_policy<T, std::enable_if_t<
    std::is_same_v<decltype(T::prefix_len), const int> &&
    std::is_same_v<decltype(T::supports_inlining), const bool> &&
    std::is_same_v<decltype(T::stores_length_explicitly), const bool>
>> : std::true_type {};

template<int kPrefixLen>
struct inlined_header_policy {
    static_assert(kPrefixLen >= 0, "Prefix length must be non-negative");
    static_assert(kPrefixLen <= 16, "Prefix length should be reasonable (<=16 bytes)");
    
    static constexpr int prefix_len = kPrefixLen;
    static constexpr bool supports_inlining = true;
    static constexpr bool stores_length_explicitly = true;
};

struct offset_only_header_policy {
    static constexpr int prefix_len = 0;
    static constexpr bool supports_inlining = false;
    static constexpr bool stores_length_explicitly = false;
};

// Split into two concrete implementations and select via std::conditional_t
template<typename LengthType, typename BufferIndexType, typename Policy>
class string_header_entry_inlined_impl {
public:
    static_assert(std::is_integral_v<LengthType>, "LengthType must be integral");
    static_assert(std::is_integral_v<BufferIndexType>, "BufferIndexType must be integral");
    static_assert(is_valid_header_policy<Policy>::value, "Invalid header policy");
    static_assert(Policy::supports_inlining, "Policy must support inlining for inlined header impl");
    
    // constexpr static int prefix_len = Policy::prefix_len;
    // constexpr static int max_inlined_len = prefix_len + sizeof(BufferIndexType);
    
    using buffer_offset_type = BufferIndexType;
    using len_type = LengthType;
    using policy_type = Policy;

    static __host__ __device__ __forceinline__ constexpr LengthType max_inlined_len() noexcept {
        return Policy::prefix_len + sizeof(BufferIndexType);
    }

    static __host__ __device__ __forceinline__ constexpr LengthType prefix_len() noexcept {
        return Policy::prefix_len;
    }

    __host__ __device__ __forceinline__ constexpr LengthType length() const noexcept {
        return len_;
    }

    __host__ __device__ __forceinline__ constexpr bool is_inlined() const noexcept {
        return length() <= max_inlined_len();
    }

    __host__ __device__ __forceinline__ const char& charAt(LengthType index) const noexcept {
        if(index < prefix_len()) {
            return prefix_[index];
        } else {
            if(is_inlined()) {
                return inlined_[index - prefix_len()];
            } else {
                GAMAX_UNREACHABLE("Accessing non-inlined data through string_header_entry");
            }
        }
    }

    __host__ __device__ __forceinline__ char& charAt(LengthType index) noexcept {
        return const_cast<char&>(static_cast<const string_header_entry_inlined_impl&>(*this).charAt(index));
    }

    __host__ __device__ __forceinline__ BufferIndexType offset() const noexcept {
        return buffer_offset_;
    }

    __host__ __device__ __forceinline__ void set_length(LengthType len) noexcept {
        len_ = len;
    }

    __host__ __device__ __forceinline__ void set_offset(BufferIndexType offset) noexcept {
        buffer_offset_ = offset;
    }

private:
    LengthType len_;
    char prefix_[Policy::prefix_len];
    union {
        char inlined_[sizeof(BufferIndexType)];
        BufferIndexType buffer_offset_;
    };
};

template<typename LengthType, typename BufferIndexType, typename Policy>
class string_header_entry_offset_impl {
public:
    static_assert(std::is_integral_v<BufferIndexType>, "BufferIndexType must be integral");
    static_assert(is_valid_header_policy<Policy>::value, "Invalid header policy");
    static_assert(!Policy::supports_inlining, "Policy must NOT support inlining for offset-only impl");
    
    using buffer_offset_type = BufferIndexType;
    using len_type = LengthType;
    using policy_type = Policy;

    __host__ __device__ __forceinline__ constexpr bool is_inlined() const noexcept {
        return false;
    }

    static __host__ __device__ __forceinline__ constexpr BufferIndexType max_inlined_len() noexcept {
        return 0;
    }

    static __host__ __device__ __forceinline__ constexpr BufferIndexType prefix_len() noexcept {
        return 0;
    }

    __host__ __device__ __forceinline__ constexpr BufferIndexType offset() const noexcept {
        return offset_;
    }

    __host__ __device__ __forceinline__ char charAt(len_type) const {
        GAMAX_UNREACHABLE("Cannot access inlined data in offset-only header");
    }

private:
    BufferIndexType offset_ {0};
};

// static_assert(sizeof(string_header_entry_impl<int32_t, gamax::size_type, offset_only_header_policy>) == 8, "Offset-only header must only contain offset");


// Convenient alias
template<typename LengthType, typename BufferIndexType, typename Policy>
using string_header_entry = std::conditional_t<Policy::supports_inlining,
    string_header_entry_inlined_impl<LengthType, BufferIndexType, Policy>,
    string_header_entry_offset_impl<LengthType, BufferIndexType, Policy>>;

static_assert(alignof(string_header_entry<int32_t, int32_t, inlined_header_policy<4>>) <= 8,
              "Header alignment should support efficient memory access");

template<typename LengthType, typename BufferIndexType, typename Policy>
class string_column_header {
public:
    using header_entry_type = string_header_entry<LengthType, BufferIndexType, Policy>;
    using len_type = LengthType;
    using policy_type = Policy;

    __host__ __device__ __forceinline__ string_column_header(header_entry_type* headers) 
        : headers_(headers) {}

    __host__ __device__ __forceinline__ LengthType length(size_type index) const noexcept {
        if constexpr (Policy::stores_length_explicitly) {
            return headers_[index].length();
        } else {
            // Arrow format: compute length from offset differences
            return headers_[index + 1].offset() - headers_[index].offset();
        }
    }

    __host__ __device__ __forceinline__ header_entry_type const& operator[](size_type index) const noexcept {
        return headers_[index];
    }

    __host__ __device__ __forceinline__ header_entry_type& operator[](size_type index) noexcept {
        return headers_[index];
    }

private:
    // for arrow-strings, there are actually (num_strings + 1) entries
    header_entry_type* headers_ {nullptr};
};

template<typename Header>
class string_flat_buffer {
public:
    using buffer_offset_type = typename Header::header_entry_type::buffer_offset_type;
    using buffer_index_type = buffer_offset_type;
    using buffer_type = char*;
    using str_idx_type = typename Header::len_type;

    __host__ __device__ string_flat_buffer(char* data_ptr, buffer_offset_type size)
        : data_ptr_(data_ptr), size_(size) {}
    
    __host__ __device__ __forceinline__ const char* data() const noexcept {
        return data_ptr_;
    }

    __host__ __device__ __forceinline__ char* data() noexcept {
        return data_ptr_;
    }

    __host__ __device__ __forceinline__ constexpr buffer_offset_type size() const noexcept {
        return size_;
    }

    __host__ __device__ __forceinline__ const char& charAt(buffer_offset_type index) const noexcept {
        return data_ptr_[index];
    }

    __host__ __device__ __forceinline__ char& charAt(buffer_offset_type index) noexcept {
        return data_ptr_[index];
    }


    __host__ __device__ __forceinline__ const char& charAt(typename Header::header_entry_type const& header, str_idx_type rel_index) const noexcept {
        if constexpr (header.max_inlined_len() == 0) {
            return charAt(header.offset() + rel_index);
        } else {
            if(header.length() <= header.max_inlined_len() || rel_index < header.prefix_len()) {
                return header.charAt(rel_index);
            } else {
                return charAt(header.offset() + rel_index - header.prefix_len());
            }
        }
    }

    __host__ __device__ __forceinline__ char& charAt(typename Header::header_entry_type& header, str_idx_type rel_index) noexcept {
        if constexpr (header.max_inlined_len() == 0) {
            return charAt(header.offset() + rel_index);
        } else {
            if(header.length() <= header.max_inlined_len() || rel_index < header.prefix_len()) {
                return header.charAt(rel_index);
            } else {
                return charAt(header.offset() + rel_index - header.prefix_len());
            }
        }
    }

private:
    char* data_ptr_ {nullptr};
    buffer_offset_type size_ {0};
};

template<class Header, class Buffer>
class string_view {
public:
    using header_entry_type = typename Header::header_entry_type;
    using buffer_type = typename Buffer::buffer_type;
    using len_type = typename Header::len_type;
    using str_idx_type = len_type;

    string_view() = delete;
    ~string_view() = default;
    string_view(string_view const& other) = default;
    string_view& operator=(string_view const& other) = default;
    string_view(string_view&& other) noexcept = default;
    string_view& operator=(string_view&& other) noexcept = default;
    
    __host__ __device__ __forceinline__
    string_view(header_entry_type const& header_impl, Buffer const& buffer_impl, len_type len)
        : header_(header_impl), buffer_(buffer_impl), len_(len) {}
    
    __host__ __device__ __forceinline__ constexpr len_type size() const noexcept {
        return len_;
    }

    __host__ __device__ __forceinline__ constexpr bool empty() const noexcept {
        return len_ == 0;
    }

    __host__ __device__ __forceinline__ const char& operator[](str_idx_type index) const noexcept {
        return buffer_.charAt(header_, index);
    }

private:
    header_entry_type const& header_;
    Buffer const& buffer_;
    len_type len_ {0};
};

template<class Header, class Buffer>
class mutable_string_view {
public:
    using header_entry_type = typename Header::header_entry_type;
    using buffer_type = typename Buffer::buffer_type;
    using len_type = typename Header::len_type;
    using str_idx_type = len_type;

    mutable_string_view() = delete;
    ~mutable_string_view() = default;
    mutable_string_view(mutable_string_view const& other) = default;
    mutable_string_view& operator=(mutable_string_view const& other) = default;
    mutable_string_view(mutable_string_view&& other) noexcept = default;
    mutable_string_view& operator=(mutable_string_view&& other) noexcept = default;
    
    __host__ __device__ __forceinline__
    mutable_string_view(header_entry_type& header_impl, Buffer& buffer_impl, len_type len)
        : header_(header_impl), buffer_(buffer_impl), len_(len) {}
    
    __host__ __device__ __forceinline__ constexpr len_type size() const noexcept {
        return len_;
    }

    __host__ __device__ __forceinline__ constexpr bool empty() const noexcept {
        return len_ == 0;
    }

    __host__ __device__ __forceinline__ char& operator[](str_idx_type index) noexcept {
        return buffer_.charAt(header_, index);
    }

    __host__ __device__ __forceinline__ header_entry_type& header() noexcept {
        return header_;
    }

private:
    header_entry_type& header_;
    Buffer& buffer_;
    len_type len_ {0};
};

template<class Header, class Buffer>
class string_column_impl {
public:
    using string_view_type = string_view<Header, Buffer>;
    using mutable_string_view_type = mutable_string_view<Header, Buffer>;
    using LengthType = typename Header::len_type;
    using BufferIndexType = typename Header::header_entry_type::buffer_offset_type;
    using header_entry_type = typename Header::header_entry_type;

    string_column_impl() = default;
    ~string_column_impl() = default;
    string_column_impl(string_column_impl const& other) = default;
    string_column_impl& operator=(string_column_impl const& other) = default;
    string_column_impl(string_column_impl&& other) noexcept = default;
    string_column_impl& operator=(string_column_impl&& other) noexcept = default;
    
    __device__
    string_column_impl(Header header_impl, Buffer buffer_impl)
        : header_impl_(header_impl), buffer_impl_(buffer_impl) {}
    
    // TODO: use the const_cast is not elegant, find a better way
    __device__
    string_column_impl(gamax::string_column_view const& col)
        : header_impl_(static_cast<typename Header::header_entry_type*>(const_cast<void*>(col.meta()))), 
          buffer_impl_(static_cast<char*>(const_cast<void*>(col.data())), col.size()) {}
    
    __device__
    string_column_impl(gamax::mutable_string_column_view const& col)
        : header_impl_(static_cast<typename Header::header_entry_type*>(col.meta())), 
          buffer_impl_(static_cast<char*>(col.data()), col.size()) {}
    
    static constexpr LengthType max_inlined_len() noexcept {
        return Header::header_entry_type::max_inlined_len();
    }

    static constexpr LengthType prefix_len() noexcept {
        return Header::header_entry_type::prefix_len();
    }

    /**
     * @brief Access string at the given index.
     * @param index Row index in the column
     * @return Read-only view of the string
     */
    __device__ __forceinline__ string_view_type viewAt(size_type index) const noexcept {
        return string_view_type(header_impl_[index], buffer_impl_, header_impl_.length(index));
    }

    /**
     * @brief Access mutable string at the given index.
     * @param index Row index in the column
     * @return Mutable view of the string
     */
    __device__ __forceinline__ mutable_string_view_type mutableViewAt(size_type index) noexcept {
        return mutable_string_view_type(header_impl_[index], buffer_impl_, header_impl_.length(index));
    }

private:
    Header header_impl_;
    Buffer buffer_impl_;
};

template<typename LengthType, typename BufferIndexType, typename Policy>
using make_string_col_impl = string_column_impl<
    string_column_header<LengthType, BufferIndexType, Policy>,
    string_flat_buffer<string_column_header<LengthType, BufferIndexType, Policy>>
>;

template<int ImplID>
struct id_to_string_impl{};

template<>
struct id_to_string_impl<GERMAN_STRING> {
    using string_col_impl = make_string_col_impl<
        int32_t,
        gamax::size_type,
        inlined_header_policy<4>>;
};

template<>
struct id_to_string_impl<ARROW_STRING> {
    using string_col_impl = make_string_col_impl<
        int32_t,
        gamax::size_type,
        offset_only_header_policy>;
};
} // namespace gamax::detail