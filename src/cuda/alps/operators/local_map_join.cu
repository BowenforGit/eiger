// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/operators/local_map_join.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuco/static_map.cuh"

#include "alps/operators/map_join_detail.cuh"   // detail::kEmptySentinel + build_compacted_output
#include "alps/error.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"

namespace alps::join {

namespace {

using gamax::size_type;

template <int CG, typename T>
std::unique_ptr<gamax::table> run(gamax::table_view left, gamax::table_view right,
                                  cudaStream_t stream, const LocalMapJoinHint& h) {
    const size_type lkey = h.left_key_idx,  lval = size_type{1} - h.left_key_idx;
    const size_type rkey = h.right_key_idx, rval = size_type{1} - h.right_key_idx;
    const size_type left_n  = left.num_rows();
    const size_type right_n = right.num_rows();

    const T* left_key_ptr  = left.get_column(lkey).data<T>();
    const T* left_val_ptr  = left.get_column(lval).data<T>();
    const T* right_key_ptr = right.get_column(rkey).data<T>();
    const T* right_val_ptr = right.get_column(rval).data<T>();
    const T  empty         = static_cast<T>(detail::kEmptySentinel);

    // Device-local unique-key map (no fabric, no MPI). Capacity = local left rows.
    // Its allocator owns a lease on the process RMM resource generation.
    using Pair = cuco::pair<T, T>;
    using Allocator = gamax::rmm_cuco_allocator<Pair>;
    using LocalMap = cuco::static_map<
        T, T, cuco::extent<std::size_t>, cuda::thread_scope_device,
        cuda::std::equal_to<T>, cuco::linear_probing<CG, cuco::default_hash_function<T>>,
        Allocator, cuco::storage<1>>;
    gamax::rmm_allocator_handle<Pair> allocator{stream};
    LocalMap map(static_cast<std::size_t>(left_n > 0 ? left_n : 1), h.lf,
                 cuco::empty_key{empty}, cuco::empty_value{empty},
                 {}, {}, {}, {}, allocator.get(), stream);

    // build: insert this rank's (key -> value) pairs (left is the unique side).
    auto pairs = thrust::make_zip_iterator(cuda::std::make_tuple(left_key_ptr, left_val_ptr));
    if (left_n > 0) map.insert_async(pairs, pairs + left_n, stream);

    // probe: found[i] = matched left value, or the -1 sentinel on a miss.
    const size_type cap = right_n > 0 ? right_n : 1;
    auto found = gamax::make_column<T>(cap, stream);
    T* found_ptr = found->template data<T>();
    if (right_n > 0)
        map.find_async(right_key_ptr, right_key_ptr + right_n, found_ptr, stream);

    // compact matches -> [found_left_value, right_key, right_value] (shared helper).
    return detail::build_compacted_output<T>(
        right_n > 0 ? found_ptr : nullptr, right_key_ptr, right_val_ptr, right_n, stream, nullptr);
}

// WIDE (int64-only, multi-payload): the map stores key -> LOCAL row index; the
// P_L left payloads and [right_key, P_R right payloads] are gathered by index into
// [left payloads, right_key, right payloads]. Mirrors shared_map_join's wide path
// minus the fabric (left is local here, index is a plain 0..left_n iota).
template <int CG>
std::unique_ptr<gamax::table> run_wide(gamax::table_view left, gamax::table_view right,
                                       cudaStream_t stream, const LocalMapJoinHint& h) {
    using T = int64_t;
    const size_type lkey = h.left_key_idx, rkey = h.right_key_idx;
    const size_type left_n  = left.num_rows();
    const size_type right_n = right.num_rows();
    const T* left_key_ptr  = left.get_column(lkey).data<T>();
    const T* right_key_ptr = right.get_column(rkey).data<T>();
    const T  empty         = static_cast<T>(detail::kEmptySentinel);

    using Pair = cuco::pair<T, T>;
    using Allocator = gamax::rmm_cuco_allocator<Pair>;
    using LocalMap = cuco::static_map<
        T, T, cuco::extent<std::size_t>, cuda::thread_scope_device,
        cuda::std::equal_to<T>, cuco::linear_probing<CG, cuco::default_hash_function<T>>,
        Allocator, cuco::storage<1>>;
    gamax::rmm_allocator_handle<Pair> allocator{stream};
    LocalMap map(static_cast<std::size_t>(left_n > 0 ? left_n : 1), h.lf,
                 cuco::empty_key{empty}, cuco::empty_value{empty},
                 {}, {}, {}, {}, allocator.get(), stream);

    // build: insert (key -> local row index 0..left_n).
    auto ids = gamax::make_column(gamax::data_type{gamax::type_id::INT64},
                                  left_n > 0 ? left_n : 1, stream);
    T* ids_ptr = ids->data<T>();
    if (left_n > 0) {
        gamax::rmm_exec_policy_handle policy{stream};
        thrust::sequence(policy.get(), ids_ptr, ids_ptr + left_n, T{0});
        auto pairs = thrust::make_zip_iterator(cuda::std::make_tuple(left_key_ptr, ids_ptr));
        map.insert_async(pairs, pairs + left_n, stream);
    }

    // probe: found[i] = matched left local index, or the -1 sentinel on a miss.
    const size_type cap = right_n > 0 ? right_n : 1;
    auto found = gamax::make_column(gamax::data_type{gamax::type_id::INT64}, cap, stream);
    T* found_ptr = found->data<T>();
    if (right_n > 0)
        map.find_async(right_key_ptr, right_key_ptr + right_n, found_ptr, stream);

    // gather: left payloads (all non-key left cols) + [right_key, non-key right cols].
    std::vector<gamax::column_view> left_views, right_views;
    for (size_type c = 0; c < left.num_columns(); ++c)
        if (c != lkey) left_views.push_back(left.get_column(c));
    right_views.push_back(right.get_column(rkey));
    for (size_type c = 0; c < right.num_columns(); ++c)
        if (c != rkey) right_views.push_back(right.get_column(c));

    return detail::build_compacted_output_wide(
        right_n > 0 ? found_ptr : nullptr,
        right_n, left_views, right_views, stream);
}

} // namespace

std::unique_ptr<gamax::table> local_map_join(gamax::table_view left, gamax::table_view right,
                                             cudaStream_t stream, const LocalMapJoinHint& hint) {
    if (hint.wide) {   // int64-only, multi-payload index+gather path
        if (left.num_columns() < 1 || right.num_columns() < 1)
            throw std::runtime_error("local_map_join (wide): left and right need >=1 column (the key; payloads optional)");
        if (hint.left_key_idx  >= static_cast<size_type>(left.num_columns()) ||
            hint.right_key_idx >= static_cast<size_type>(right.num_columns()))
            throw std::runtime_error("local_map_join (wide): key_idx out of range");
        auto all_i64 = [](gamax::table_view t) {
            for (size_type c = 0; c < t.num_columns(); ++c)
                if (t.get_column(c).type().id() != gamax::type_id::INT64) return false;
            return true;
        };
        if (!all_i64(left) || !all_i64(right))
            throw std::runtime_error("local_map_join (wide): all columns must be INT64");
        switch (hint.cg) {
            case 1: return run_wide<1>(left, right, stream, hint);
            case 2: return run_wide<2>(left, right, stream, hint);
            case 4: return run_wide<4>(left, right, stream, hint);
            default: throw std::runtime_error("local_map_join: hint.cg must be 1, 2, or 4");
        }
    }

    if (left.num_columns() != 2 || right.num_columns() != 2)
        throw std::runtime_error("local_map_join: left and right must have exactly 2 columns");
    if (hint.left_key_idx > 1 || hint.right_key_idx > 1)
        throw std::runtime_error("local_map_join: key_idx must be 0 or 1");

    const gamax::type_id id = left.get_column(hint.left_key_idx).type().id();
    auto same = [&](const gamax::column_view& c) { return c.type().id() == id; };
    if (!same(left.get_column(size_type{1} - hint.left_key_idx)) ||
        !same(right.get_column(hint.right_key_idx)) ||
        !same(right.get_column(size_type{1} - hint.right_key_idx)))
        throw std::runtime_error("local_map_join: all key/value columns must share one type");
    if (id != gamax::type_id::INT32 && id != gamax::type_id::INT64)
        throw std::runtime_error("local_map_join: only INT32 or INT64 columns are supported");

    const bool i64 = (id == gamax::type_id::INT64);
    switch (hint.cg) {
        case 1: return i64 ? run<1, int64_t>(left, right, stream, hint)
                           : run<1, int32_t>(left, right, stream, hint);
        case 2: return i64 ? run<2, int64_t>(left, right, stream, hint)
                           : run<2, int32_t>(left, right, stream, hint);
        case 4: return i64 ? run<4, int64_t>(left, right, stream, hint)
                           : run<4, int32_t>(left, right, stream, hint);
        default: throw std::runtime_error("local_map_join: hint.cg must be 1, 2, or 4");
    }
}

} // namespace alps::join
