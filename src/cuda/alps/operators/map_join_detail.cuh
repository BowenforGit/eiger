// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Shared internals for the static_map-based PK-FK joins (shared_map_join over a
// fabric-striped map, local_map_join over a device-local map). Both share one
// post-find compaction, in two flavours:
//   NARROW (build_compacted_output<T>): the slot stores the left value inline, so
//     `found` IS the value; zip (found, right_key, right_value), keep matches, emit
//     [found_left_value, right_key, right_value] (3 cols). int32 or int64.
//   WIDE (build_compacted_output_wide): the slot stores a left-row INDEX (int64;
//     padded-global for shared_map, local for local_map_join), so we select the
//     matched (left_idx, right_pos) pairs then GATHER each left payload by left_idx
//     and each right column by right_pos -> [left payloads, right_key, right
//     payloads]. gather reads the left views over the fabric when they point there.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "alps/error.hpp"
#include "memory/column.hpp"
#include "memory/device_buffer.hpp"
#include "memory/table.hpp"
#include "utils/transform.hpp"   // gamax::util::device::gather (wide-mode column gather)

namespace alps::join::detail {

// Empty sentinel: -1 works for both int32 and int64 (all real keys/values >= 0).
// Used as the cuco empty_key/empty_value AND as the "miss" marker in the compaction.
constexpr long long kEmptySentinel = -1;

// Predicate for DeviceSelect::If over a zip of (found, right_key, right_value):
// keep the row iff the looked-up left value is not the empty sentinel (a match).
// Templated on the tuple type so it is agnostic to thrust vs cuda::std tuples.
template <typename T>
struct IsMatch {
    T sentinel;
    template <typename Tuple>
    __host__ __device__ bool operator()(const Tuple& t) const {
        return thrust::get<0>(t) != sentinel;
    }
};

// Compact the find results into a 3-column table [found_left_value, right_key,
// right_value], keeping only matches (found != -1). `found` is the per-right-row
// lookup result (may be nullptr iff right_n == 0). The three output columns are
// over-allocated to the right_n upper bound (>=1, never 0-byte) and truncated to
// the match count via set_rows. cub scratch is allocated OUTSIDE the optional
// timing window (cudaMallocAsync pool growth would otherwise perturb it); when
// `compact_ms` is non-null it receives the CUDA-event time of just the select +
// count copy. num_items / the count word are 64-bit (size_type), so probe sides
// and match counts above 2^31 are handled (cuco find + cub are 64-bit-clean).
template <typename T>
std::unique_ptr<gamax::table> build_compacted_output(
    const T* found, const T* right_key, const T* right_val,
    gamax::size_type right_n, cudaStream_t stream, float* compact_ms = nullptr) {
    using gamax::size_type;
    const size_type cap = right_n > 0 ? right_n : 1;
    auto out_val = gamax::make_column<T>(cap, stream);
    auto out_rk  = gamax::make_column<T>(cap, stream);
    auto out_rv  = gamax::make_column<T>(cap, stream);

    size_type num = 0;
    if (right_n > 0) {
        auto in_zip = thrust::make_zip_iterator(thrust::make_tuple(found, right_key, right_val));
        auto out_zip = thrust::make_zip_iterator(thrust::make_tuple(
            out_val->template data<T>(), out_rk->template data<T>(), out_rv->template data<T>()));
        IsMatch<T> pred{static_cast<T>(kEmptySentinel)};

        gamax::device_buffer num_storage(sizeof(size_type), stream);
        auto* d_num = static_cast<size_type*>(num_storage.data());
        size_t temp_bytes = 0;
        cub::DeviceSelect::If(nullptr, temp_bytes, in_zip, out_zip, d_num, right_n, pred, stream);
        gamax::device_buffer temp_storage(temp_bytes, stream);
        void* d_temp = temp_storage.data();

        cudaEvent_t e0{}, e1{};
        if (compact_ms) { ALPS_CUDACHECK(cudaEventCreate(&e0)); ALPS_CUDACHECK(cudaEventCreate(&e1));
                          ALPS_CUDACHECK(cudaEventRecord(e0, stream)); }
        size_type h_num = 0;
        cub::DeviceSelect::If(d_temp, temp_bytes, in_zip, out_zip, d_num, right_n, pred, stream);
        ALPS_CUDACHECK(cudaMemcpyAsync(&h_num, d_num, sizeof(size_type), cudaMemcpyDeviceToHost, stream));
        if (compact_ms) { ALPS_CUDACHECK(cudaEventRecord(e1, stream)); ALPS_CUDACHECK(cudaEventSynchronize(e1));
                          ALPS_CUDACHECK(cudaEventElapsedTime(compact_ms, e0, e1));
                          ALPS_CUDACHECK(cudaEventDestroy(e0)); ALPS_CUDACHECK(cudaEventDestroy(e1)); }
        else ALPS_CUDACHECK(cudaStreamSynchronize(stream));   // need h_num below
        num = h_num;
    }

    std::vector<std::unique_ptr<gamax::column>> cols;
    cols.reserve(3);
    cols.push_back(std::move(out_val));
    cols.push_back(std::move(out_rk));
    cols.push_back(std::move(out_rv));
    auto out = std::make_unique<gamax::table>(std::move(cols));
    out->set_rows(num);
    return out;
}

// Wide-mode compaction (multi-payload). `found_idx[i]` = the matched left-row INDEX
// for right row i, or the -1 sentinel on a miss (int64; padded-global for
// shared_map_join, local for local_map_join). Phase 1 selects the matched
// (left_idx, right_pos) pairs in one cub pass; phase 2 gathers each column of
// `left_views` by left_idx and each column of `right_views` by right_pos. The
// caller orders `right_views` as [right_key, right_payloads...], so the output
// schema is [left payloads..., right_key, right_payloads...] (left key dropped).
// gather is agnostic to whether a left view points at local or fabric memory --
// for shared_map the left views are FabricStripedRegion bases, so those gathers are
// remote reads over NVLink. `compact_ms` times the cub select; `gather_ms` times the
// multi-column gather (the fabric-read materialization cost for shared_map). Not a
// template (works on type-dispatched column_views) so it is `inline` to avoid ODR
// clashes across the TUs that include this header.
inline std::unique_ptr<gamax::table> build_compacted_output_wide(
    const int64_t* found_idx, gamax::size_type right_n,
    const std::vector<gamax::column_view>& left_views,
    const std::vector<gamax::column_view>& right_views,
    cudaStream_t stream, float* compact_ms = nullptr, float* gather_ms = nullptr) {
    using gamax::size_type;
    const gamax::data_type idx_type{gamax::type_id::INT64};

    // ---- phase 1: select matched (left_idx, right_pos) pairs (found_idx != -1) ----
    const size_type cap = right_n > 0 ? right_n : 1;
    auto left_idx  = gamax::make_column(idx_type, cap, stream);
    auto right_pos = gamax::make_column(idx_type, cap, stream);
    int64_t* left_idx_ptr  = left_idx->data<int64_t>();
    int64_t* right_pos_ptr = right_pos->data<int64_t>();

    size_type num = 0;
    if (right_n > 0) {
        auto pos_it  = thrust::make_counting_iterator<int64_t>(0);
        auto in_zip  = thrust::make_zip_iterator(thrust::make_tuple(found_idx, pos_it));
        auto out_zip = thrust::make_zip_iterator(thrust::make_tuple(left_idx_ptr, right_pos_ptr));
        IsMatch<int64_t> pred{static_cast<int64_t>(kEmptySentinel)};

        gamax::device_buffer num_storage(sizeof(size_type), stream);
        auto* d_num = static_cast<size_type*>(num_storage.data());
        size_t temp_bytes = 0;
        cub::DeviceSelect::If(nullptr, temp_bytes, in_zip, out_zip, d_num, right_n, pred, stream);
        gamax::device_buffer temp_storage(temp_bytes, stream);
        void* d_temp = temp_storage.data();

        cudaEvent_t e0{}, e1{};
        if (compact_ms) { ALPS_CUDACHECK(cudaEventCreate(&e0)); ALPS_CUDACHECK(cudaEventCreate(&e1));
                          ALPS_CUDACHECK(cudaEventRecord(e0, stream)); }
        size_type h_num = 0;
        cub::DeviceSelect::If(d_temp, temp_bytes, in_zip, out_zip, d_num, right_n, pred, stream);
        ALPS_CUDACHECK(cudaMemcpyAsync(&h_num, d_num, sizeof(size_type), cudaMemcpyDeviceToHost, stream));
        if (compact_ms) { ALPS_CUDACHECK(cudaEventRecord(e1, stream)); ALPS_CUDACHECK(cudaEventSynchronize(e1));
                          ALPS_CUDACHECK(cudaEventElapsedTime(compact_ms, e0, e1));
                          ALPS_CUDACHECK(cudaEventDestroy(e0)); ALPS_CUDACHECK(cudaEventDestroy(e1)); }
        else ALPS_CUDACHECK(cudaStreamSynchronize(stream));   // need h_num below
        num = h_num;
    }

    // ---- phase 2: gather each output column by the compacted indices ----
    gamax::column_view left_idx_view (left_idx_ptr,  num, idx_type);
    gamax::column_view right_pos_view(right_pos_ptr, num, idx_type);
    std::vector<std::unique_ptr<gamax::column>> cols;
    cols.reserve(left_views.size() + right_views.size());

    cudaEvent_t g0{}, g1{};
    if (gather_ms) { ALPS_CUDACHECK(cudaEventCreate(&g0)); ALPS_CUDACHECK(cudaEventCreate(&g1));
                     ALPS_CUDACHECK(cudaEventRecord(g0, stream)); }
    for (const auto& lv : left_views)  cols.push_back(gamax::util::device::gather(left_idx_view,  lv, stream));
    for (const auto& rv : right_views) cols.push_back(gamax::util::device::gather(right_pos_view, rv, stream));
    if (gather_ms) { ALPS_CUDACHECK(cudaEventRecord(g1, stream)); ALPS_CUDACHECK(cudaEventSynchronize(g1));
                     ALPS_CUDACHECK(cudaEventElapsedTime(gather_ms, g0, g1));
                     ALPS_CUDACHECK(cudaEventDestroy(g0)); ALPS_CUDACHECK(cudaEventDestroy(g1)); }

    auto out = std::make_unique<gamax::table>(std::move(cols));
    out->set_rows(num);
    return out;
}

} // namespace alps::join::detail
