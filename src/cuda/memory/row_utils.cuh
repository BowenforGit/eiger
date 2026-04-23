#pragma once

#include <cuco/hash_functions.cuh>
#include <cstddef>
#include <cuda_runtime_api.h>

#include "memory/table.hpp"
#include "types/type_dispatcher.hpp"
#include "string/string_column.hpp"
#include "string/string_impl.cuh"

namespace gamax {

using arrow_string_col_impl = gamax::detail::id_to_string_impl<ARROW_STRING>::string_col_impl;

struct element_equal_functor {
    template<typename T, std::enable_if_t<!std::is_same_v<T, gamax::string_type_tag>, int> = 0>
    __device__ bool operator()(gamax::column_view& col, gamax::size_type lhs_idx, gamax::size_type rhs_idx) const {
        const T* typed_data = col.data<T>();
        return typed_data[lhs_idx] == typed_data[rhs_idx];
    }

    template<typename T, std::enable_if_t<std::is_same_v<T, gamax::string_type_tag>, int> = 0>
    __device__ bool operator()(gamax::column_view& col, gamax::size_type lhs_idx, gamax::size_type rhs_idx) const {
        gamax::string_column_view str_col(col, ARROW_STRING);
        arrow_string_col_impl str_impl(str_col);
        
        auto lhs_view = str_impl.viewAt(lhs_idx);
        auto rhs_view = str_impl.viewAt(rhs_idx);
        
        return lhs_view.compare(rhs_view);
    }
};

struct element_hash_functor {
    template<typename T, std::enable_if_t<!std::is_same_v<T, gamax::string_type_tag>, int> = 0>
    __device__ std::size_t operator()(gamax::column_view& col, gamax::size_type idx) const {
        const T* typed_data = col.data<T>();
        return cuco::default_hash_function<T>{}(typed_data[idx]);
    }

    template<typename T, std::enable_if_t<std::is_same_v<T, gamax::string_type_tag>, int> = 0>
    __device__ std::size_t operator()(gamax::column_view& col, gamax::size_type idx) const {
        gamax::string_column_view str_col(col, ARROW_STRING);
        arrow_string_col_impl str_impl(str_col);

        auto str_view = str_impl.viewAt(idx);

        cuco::default_hash_function<char> hasher{};
        return hasher.compute_hash(
            reinterpret_cast<std::byte const*>(str_view.data()),
            str_view.size());
    }
};

// ============================================================================
// Row-level functors supporting both numeric and string types
// ============================================================================

// Check if two rows from the same table are equal (supports STRING columns)
struct row_equal {
    gamax::experimental::table_device_view _keys;

    __host__ row_equal(gamax::experimental::table_device_view keys) : _keys{keys} {}

    __device__ bool operator()(gamax::size_type lhs, gamax::size_type rhs) const
    {
        for(gamax::size_type col = 0; col < _keys.num_columns(); col++) {
            auto col_view = _keys.get_column(col);
            bool equal = gamax::type_dispatcher_all(col_view.type(), element_equal_functor{}, col_view, lhs, rhs);
            if(!equal) return false;
        }
        return true;
    }
};

// Multi-column row hasher using hash combining (supports STRING columns)
struct row_hasher {
    gamax::experimental::table_device_view _keys;

    __host__ row_hasher(gamax::experimental::table_device_view keys) : _keys{keys} {}

    __device__ std::size_t operator()(gamax::size_type index) const
    {
        std::size_t hash = 0;
        for(gamax::size_type col = 0; col < _keys.num_columns(); col++) {
            auto col_view = _keys.get_column(col);
            std::size_t col_hash = gamax::type_dispatcher_all(col_view.type(), element_hash_functor{}, col_view, index);
            
            // Combine hashes using a variant of FNV-1a combining
            hash ^= col_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// Row-copy helpers used to live here; they were numeric-only (`type_dispatcher`
// has no STRING case) and their only caller — the hash-groupby insert kernel —
// now records a leader-tid per new group and materialises key columns via
// `join::detail::simple_materialize` after the kernel returns, so STRING keys
// work uniformly. See `operators/groupby/groupby_hash.cu::compute_groupid_kernel`.

} // namespace gamax