// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <cub/device/device_run_length_encode.cuh>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include "utils/utils.cuh"
#include "utils/cub_wrapper.cuh"
#include "utils/copying.hpp"
#include "utils/transform.hpp"
#include "compress/dictionary_encode.hpp"
#include "compress/multikey_compress.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "operators/join/materialize.hpp"
#include "operators/sort/sort.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax::groupby::sort {

namespace {

template<typename F, typename ValT>
struct reduce_by_key_aggregator_helper {
    template<typename KeyT>
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        gamax::mutable_column_view result_key,
        gamax::mutable_column_view result_val,
        gamax::size_type* d_num_groups,
        gamax::size_type* permutation_idx,
        cudaStream_t stream) 
    {
        using AggT = typename F::OutputType;
        
        auto num_items = key.size();
        
        if(permutation_idx != nullptr) {
            auto val_itr = thrust::make_permutation_iterator(val.data<ValT>(), permutation_idx);
            CALL_CUB_FUNC(stream, cub::DeviceReduce::ReduceByKey, key.data<KeyT>(), result_key.data<KeyT>(), val_itr, result_val.data<AggT>(), d_num_groups, F{}, num_items, stream)
        } else {
            CALL_CUB_FUNC(stream, cub::DeviceReduce::ReduceByKey, key.data<KeyT>(), result_key.data<KeyT>(), val.data<ValT>(), result_val.data<AggT>(), d_num_groups, F{}, num_items, stream)
        }
    }

    template<typename KeyT>
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        const gamax::size_type num_groups,
        gamax::mutable_column_view result_val,
        gamax::size_type* permutation_idx,
        cudaStream_t stream) 
    {
        using AggT = typename F::OutputType;
        
        auto num_items = key.size();
        auto result_key_ptr = gamax::make_column(key.type(), num_groups, stream);
        auto d_num_groups = gamax::make_column(gamax::data_type{size_type_id}, 1, stream);

        if(permutation_idx != nullptr) {
            auto val_itr = thrust::make_permutation_iterator(val.data<ValT>(), permutation_idx);
            CALL_CUB_FUNC(stream, cub::DeviceReduce::ReduceByKey, key.data<KeyT>(), result_key_ptr->data<KeyT>(), val_itr, result_val.data<AggT>(), d_num_groups->data<gamax::size_type>(), F{}, num_items, stream)
        } else {
            CALL_CUB_FUNC(stream, cub::DeviceReduce::ReduceByKey, key.data<KeyT>(), result_key_ptr->data<KeyT>(), val.data<ValT>(), result_val.data<AggT>(), d_num_groups->data<gamax::size_type>(), F{}, num_items, stream)
        }
    }
};

template<gamax::groupby::aggregation_id agg_kind>
struct dispatch_key_type_helper {
    template<typename ValT>
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        gamax::mutable_column_view result_key,
        gamax::mutable_column_view result_val,
        gamax::size_type* d_num_groups,
        gamax::size_type* permutation_idx,
        cudaStream_t stream) 
    {
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, ValT>::F;
        
        gamax::type_dispatcher(
            key.type(), 
            reduce_by_key_aggregator_helper<F, ValT>{}, 
            key, 
            val, 
            result_key, 
            result_val, 
            d_num_groups,
            permutation_idx,
            stream
        );
    }

    template<typename ValT>
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        const gamax::size_type num_groups,
        gamax::mutable_column_view result_val,
        gamax::size_type* permutation_idx,
        cudaStream_t stream) 
    {
        using F = typename agg_fn_mapper<agg_kind, ValT>::F;

        gamax::type_dispatcher(
            key.type(), 
            reduce_by_key_aggregator_helper<F, ValT>{}, 
            key, 
            val, 
            num_groups,
            result_val,
            permutation_idx,
            stream
        );
    }
};

template<gamax::groupby::aggregation_id agg_kind>
struct dispatch_val_type_helper {
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        gamax::mutable_column_view result_key,
        gamax::mutable_column_view result_val,
        gamax::size_type* d_num_groups,
        gamax::size_type* permutation_idx,
        cudaStream_t stream) 
    {
    
        gamax::type_dispatcher(
            val.type(), 
            dispatch_key_type_helper<agg_kind>{}, 
            key, 
            val, 
            result_key, 
            result_val, 
            d_num_groups,
            permutation_idx,
            stream
        );
    }

    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        const gamax::size_type num_groups,
        gamax::mutable_column_view result_val,
        gamax::size_type* permutation_idx,
        cudaStream_t stream) 
    {
        gamax::type_dispatcher(
            val.type(), 
            dispatch_key_type_helper<agg_kind>{}, 
            key, 
            val, 
            num_groups,
            result_val,
            permutation_idx,
            stream
        );
    }
};

DISPATCH_AGGREGATE(dispatch_val_type_helper, __host__)

void aggregate(
    gamax::groupby::aggregation_id agg_kind, 
    gamax::column_view key, 
    gamax::column_view val, 
    gamax::mutable_column_view result_key,
    gamax::mutable_column_view result_val,
    gamax::size_type& num_groups,
    gamax::size_type* permutation_idx,
    cudaStream_t stream)
{
    auto d_num_groups = gamax::make_column(gamax::data_type{size_type_id}, 1, stream);
    dispatch_aggregate(agg_kind, key, val, result_key, result_val, d_num_groups->data<gamax::size_type>(), permutation_idx, stream);
    cudaMemcpyAsync(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

void aggregate(
    gamax::groupby::aggregation_id agg_kind, 
    gamax::column_view key, 
    gamax::column_view val, 
    const gamax::size_type num_groups,
    gamax::mutable_column_view result_val,
    gamax::size_type* permutation_idx,
    cudaStream_t stream)
{
    dispatch_aggregate(agg_kind, key, val, num_groups, result_val, permutation_idx, stream);
    cudaStreamSynchronize(stream);
}

void check_solvable(gamax::table_view const& key, gamax::table_view const& val, std::vector<gamax::groupby::aggregation_request>& agg_requests) {
    if(key.num_columns() > 1) {
        throw std::runtime_error("Only single-column group keys are supported");
    }
    
    if(key.num_columns() == 0) {
        throw std::runtime_error("Aggregation only is not supported yet");
    }

    if(agg_requests.size() == 0) {
        throw std::runtime_error("No aggregation function is specified");
    }
}

void extract_groupby_hint(
    gamax::groupby::groupby_hint& hint, 
    gamax::size_type input_num_rows, 
    gamax::data_type dtype, 
    gamax::size_type& estimated_num_groups, 
    int& begin_bit, 
    int& end_bit) 
{
    begin_bit = 0;
    end_bit = 8 * gamax::size_of(dtype);
    estimated_num_groups = std::min(input_num_rows, hint.groups_upper_bound);
    
    if(hint.zero_based_ids) {
        end_bit = [&](gamax::size_type v) {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v++;
            return 8 * gamax::size_of(dtype) - __builtin_clz(v);
        } (estimated_num_groups);
    }
}

// ── distinct helpers ─────────────────────────────────────────────
//
// Case 1 (single-col input): sort the column, then run
// `cub::DeviceRunLengthEncode::Encode` with the unique-values output
// connected to a pre-allocated column and the per-run counts discarded.
// `d_num_runs_out` returns the count of unique values; the unique
// column is trimmed and returned.
//
// Templated on key type T so we can hand CUB typed pointers. Used via
// `type_dispatcher(key.type(), case1_dispatcher{}, ...)` so that every
// numeric key type the engine handles works without an explicit
// switch.
struct case1_dispatcher {
    template<typename T>
    std::unique_ptr<gamax::table> operator()(
        gamax::column_view sorted_key,
        cudaStream_t stream) const
    {
        const auto n = sorted_key.size();

        // Allocate the unique-values output at the worst-case size
        // (every row distinct) and trim after the RLE writes the
        // actual `num_runs`.
        auto unique_col = gamax::make_column<T>(n, stream);
        auto d_num_runs = gamax::make_idx_column(1, stream);

        std::size_t tmp_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(
            nullptr, tmp_bytes,
            sorted_key.template data<T>(),
            unique_col->template data<T>(),
            thrust::make_discard_iterator(),
            d_num_runs->data<gamax::size_type>(),
            n, stream);
        auto tmp = std::make_unique<gamax::device_buffer>(tmp_bytes, stream);
        cub::DeviceRunLengthEncode::Encode(
            tmp->data(), tmp_bytes,
            sorted_key.template data<T>(),
            unique_col->template data<T>(),
            thrust::make_discard_iterator(),
            d_num_runs->data<gamax::size_type>(),
            n, stream);

        gamax::size_type h_num_runs = 0;
        cudaMemcpyAsync(&h_num_runs,
                        d_num_runs->data<gamax::size_type>(),
                        sizeof(gamax::size_type),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        unique_col->set_size(h_num_runs);

        std::vector<std::unique_ptr<gamax::column>> cols;
        cols.push_back(std::move(unique_col));
        return std::make_unique<gamax::table>(std::move(cols));
    }
};

// Case 2 helper: RLE the SKF-compressed (uint64) sorted key. The unique
// values are discarded (we'll materialize original keys via the
// leader-tid pattern instead); the per-run counts are kept so we can
// turn them into the leader-row indices via an exclusive prefix sum.
// Returns the host-side `num_runs`.
inline gamax::size_type rle_counts_only(
    gamax::column_view sorted_fk,
    gamax::mutable_column_view counts_out,
    cudaStream_t stream)
{
    const auto n = sorted_fk.size();
    auto d_num_runs = gamax::make_idx_column(1, stream);

    std::size_t tmp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(
        nullptr, tmp_bytes,
        sorted_fk.data<std::uint64_t>(),
        thrust::make_discard_iterator(),
        counts_out.data<gamax::size_type>(),
        d_num_runs->data<gamax::size_type>(),
        n, stream);
    auto tmp = std::make_unique<gamax::device_buffer>(tmp_bytes, stream);
    cub::DeviceRunLengthEncode::Encode(
        tmp->data(), tmp_bytes,
        sorted_fk.data<std::uint64_t>(),
        thrust::make_discard_iterator(),
        counts_out.data<gamax::size_type>(),
        d_num_runs->data<gamax::size_type>(),
        n, stream);

    gamax::size_type h_num_runs = 0;
    cudaMemcpyAsync(&h_num_runs,
                    d_num_runs->data<gamax::size_type>(),
                    sizeof(gamax::size_type),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h_num_runs;
}

} // unnamed namespace

std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream) 
{
    check_solvable(key, val, agg_requests);

    auto input_num_rows = key.num_rows();
    auto key_type = key.get_column(0).type();
    
    gamax::size_type estimated_num_groups;
    int begin_bit, end_bit;
    extract_groupby_hint(hint, input_num_rows, key_type, estimated_num_groups, begin_bit, end_bit);

    auto first_val_col_id = agg_requests[0].column_index;
    auto [sorted_key, sorted_val_0] = gamax::sort::sort_column_pair(
        key.get_column(0), val.get_column(first_val_col_id), stream,
        begin_bit, end_bit, gamax::sort_order::ASCENDING, nullptr, 0);
    
    gamax::size_type num_groups = 0;
    std::vector<std::unique_ptr<gamax::column>> final_columns;

    std::unique_ptr<gamax::column> encoded_key = nullptr;

    bool grouped = false;
    for(auto& req: agg_requests) {
        auto i = req.column_index;
        auto agg_kind = req.agg_kind;
        std::unique_ptr<column> sorted_val = nullptr;
        auto val_type = val.get_column(i).type();
        auto agg_output_type = gamax::groupby::agg_output_data_type(agg_kind, val_type);
        
        if(!grouped) {
            sorted_val = std::move(sorted_val_0);
            auto unique_keys = gamax::make_column(key_type, estimated_num_groups, stream);
            auto agg_val = gamax::make_column(agg_output_type, estimated_num_groups, stream);
            
            aggregate(agg_kind, sorted_key->view(), sorted_val->view(), unique_keys->mutable_view(), agg_val->mutable_view(), num_groups, nullptr, stream);

            unique_keys->set_size(num_groups);
            agg_val->set_size(num_groups);

            final_columns.push_back(std::move(unique_keys));
            final_columns.push_back(std::move(agg_val));

            grouped = true;
        }
        else {
            if(hint.optimize_sort) {
                if(encoded_key.get() == nullptr) {
                    encoded_key = gamax::compress::dictionary_encode(
                        key.get_column(0), 
                        // *(final_columns.front()), 
                        final_columns.front()->view(),
                        stream,
                        data_type{type_id::INT32});
                }
                auto end_bit = [&](gamax::size_type v) {
                    v--;
                    v |= v >> 1;
                    v |= v >> 2;
                    v |= v >> 4;
                    v |= v >> 8;
                    v |= v >> 16;
                    v++;
                    return 8 * gamax::size_of(encoded_key->type()) - __builtin_clz(v);
                } (num_groups);
                std::tie(std::ignore, sorted_val) = gamax::sort::sort_column_pair(
                    encoded_key->view(), val.get_column(i), stream,
                    0, end_bit, gamax::sort_order::ASCENDING, nullptr, 0);
            } else {
                std::tie(std::ignore, sorted_val) = gamax::sort::sort_column_pair(
                    key.get_column(0), val.get_column(i), stream,
                    begin_bit, end_bit, gamax::sort_order::ASCENDING, nullptr, 0);
            }

            auto agg_val = gamax::make_column(agg_output_type, num_groups, stream);
            aggregate(agg_kind, sorted_key->view(), sorted_val->view(), num_groups, agg_val->mutable_view(), nullptr, stream);
            final_columns.push_back(std::move(agg_val));
        }
    }

    cudaStreamSynchronize(stream);

    return std::make_unique<gamax::table>(std::move(final_columns));
}

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table_view key,
    gamax::table_view val,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    // std::cout << "groupby_gfur" << std::endl;
    check_solvable(key, val, agg_requests);

    auto input_num_rows = key.num_rows();
    auto key_type = key.get_column(0).type();
    
    gamax::size_type estimated_num_groups;
    int begin_bit, end_bit;
    extract_groupby_hint(hint, input_num_rows, key_type, estimated_num_groups, begin_bit, end_bit);

    auto [sorted_key, idx_col] = gamax::sort::argsort(
        key.get_column(0), stream, begin_bit, end_bit,
        gamax::sort_order::ASCENDING, nullptr, 0);
    
    gamax::size_type num_groups = 0;
    std::vector<std::unique_ptr<gamax::column>> final_columns;

    bool grouped = false;
    for(auto& req: agg_requests) {
        auto i = req.column_index;
        auto agg_kind = req.agg_kind;

        auto val_type = val.get_column(i).type();
        auto agg_output_type = gamax::groupby::agg_output_data_type(agg_kind, val_type);
        
        if(!grouped) {
            auto agg_val = gamax::make_column(agg_output_type, estimated_num_groups, stream);
            auto unique_keys = gamax::make_column(key_type, estimated_num_groups, stream);
            aggregate(agg_kind, sorted_key->view(), val.get_column(i), unique_keys->mutable_view(), agg_val->mutable_view(), num_groups, idx_col->data<gamax::size_type>(), stream);
            unique_keys->set_size(num_groups);
            agg_val->set_size(num_groups);
            final_columns.push_back(std::move(unique_keys));
            final_columns.push_back(std::move(agg_val));

            grouped = true;
        } else {
            auto agg_val = gamax::make_column(agg_output_type, num_groups, stream);
            aggregate(agg_kind, sorted_key->view(), val.get_column(i), num_groups, agg_val->mutable_view(), idx_col->data<gamax::size_type>(), stream);
            final_columns.push_back(std::move(agg_val));
        }
    }

    cudaStreamSynchronize(stream);

    return std::make_unique<gamax::table>(std::move(final_columns));
}

// Sort-based distinct. See groupby_sort.hpp for scope / contract.
//
// Branching (single-col-first):
//   * key.num_columns() == 1   → Case 1 directly
//   * hint.enable_skf set      → try multikey_compress; success → Case 2
//                                  failure → throw (multi-col + compress
//                                  declined means we can't reduce to a
//                                  single key)
//   * otherwise                → throw (sort distinct is single-key only)
//
// STRING keys are rejected up front with an actionable message —
// `sort_column` / `sort_column_pair` use CUB radix sort which doesn't
// accept STRING anyway.
std::unique_ptr<gamax::table> distinct(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    if (key.num_columns() == 0) {
        throw std::runtime_error(
            "sort::distinct: must have at least one key column");
    }

    // STRING guard. Reject early — the underlying radix sort can't
    // handle string columns, and falling through would surface a less
    // specific error later.
    for (gamax::size_type i = 0; i < key.num_columns(); ++i) {
        if (key.get_column(i).type().id() == gamax::type_id::STRING) {
            throw std::runtime_error(
                "sort::distinct: STRING keys not supported; use HASH");
        }
    }

    const auto n = key.num_rows();

    // ── Case 1: truly a single key column ──────────────────────────
    if (key.num_columns() == 1) {
        auto sorted_key = gamax::sort::sort_column(
            key.get_column(0),
            stream,
            /*begin_bit=*/0, /*end_bit=*/-1,
            gamax::sort_order::ASCENDING,
            /*d_temp_storage=*/nullptr, /*temp_storage_bytes=*/0);

        return gamax::type_dispatcher(
            sorted_key->type(),
            case1_dispatcher{},
            sorted_key->view(),
            stream);
    }

    // ── Case 2: multi-col + SKF ────────────────────────────────────
    // SKF must be enabled; otherwise the algorithm can't reduce to a
    // single-key sort.
    if (!hint.enable_skf) {
        throw std::runtime_error(
            "sort::distinct: only single-col keys are supported "
            "natively; set hint.enable_skf to enable multi-col SKF "
            "fusion (or use HASH)");
    }

    int compressed_bits = 0;
    gamax::compress::multikey_compress_options skf_opts;
    skf_opts.out_total_bits = &compressed_bits;
    auto fk = gamax::compress::multikey_compress(key, stream, skf_opts);
    if (!fk) {
        throw std::runtime_error(
            "sort::distinct: multi-col key didn't compress under SKF "
            "(bit budget or unsupported type); use HASH");
    }

    // Per-row tids [0, n), so the (fk, tid) sort gives us the original
    // row index of every key after sorting.
    auto tids = gamax::make_sequence<gamax::size_type>(
        n, stream, /*start=*/gamax::size_type{0}, /*step=*/gamax::size_type{1});

    auto [sorted_fk, sorted_tid] = gamax::sort::sort_column_pair(
        fk->view(), tids->view(),
        stream,
        /*begin_bit=*/0, /*end_bit=*/-1,
        gamax::sort_order::ASCENDING,
        /*d_temp_storage=*/nullptr, /*temp_storage_bytes=*/0);

    // RLE on the sorted fused key. Discard unique values (we materialize
    // original-key rows via leader-tid instead); keep per-group counts
    // so we can do an exclusive prefix sum into leader-row indices.
    auto counts = gamax::make_column(
        gamax::data_type{size_type_id}, n, stream);
    const gamax::size_type num_runs = rle_counts_only(
        sorted_fk->view(), counts->mutable_view(), stream);

    // Fast path: every row was unique → no dedup work needed; the
    // output is a column-wise deep-copy of the original input table
    // (distinct's output row order is unspecified, so we don't need
    // the sorted permutation here).
    if (num_runs == n) {
        std::vector<std::unique_ptr<gamax::column>> cols;
        cols.reserve(key.num_columns());
        for (gamax::size_type i = 0; i < key.num_columns(); ++i) {
            cols.push_back(gamax::copy_column(key.get_column(i), stream));
        }
        return std::make_unique<gamax::table>(std::move(cols));
    }

    // Slow path: trim counts to `num_runs`, exclusive-sum into leader
    // positions in the sorted order, gather original row indices from
    // `sorted_tid` at those positions, then materialize the original
    // multi-col keys via the leader-tid map.
    counts->set_size(num_runs);
    auto leader_idx = gamax::util::device::prefix_sum(
        counts->view(),
        gamax::prefix_sum_options::EXCLUSIVE,
        gamax::data_type{size_type_id},
        stream);

    auto leader_tid = gamax::util::device::gather(
        leader_idx->view(),
        sorted_tid->view(),
        stream);

    std::vector<std::unique_ptr<gamax::column>> cols;
    cols.reserve(key.num_columns());
    for (gamax::size_type i = 0; i < key.num_columns(); ++i) {
        cols.push_back(gamax::join::detail::simple_materialize(
            key.get_column(i), leader_tid->view(), stream));
    }
    cudaStreamSynchronize(stream);
    return std::make_unique<gamax::table>(std::move(cols));
}

} // namespace gamax::groupby::sort
