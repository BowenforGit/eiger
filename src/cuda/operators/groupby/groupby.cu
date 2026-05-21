#include "groupby.hpp"

#include <algorithm>
#include <cuda.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/groupby_hash.hpp"
#include "operators/groupby/groupby_partition.hpp"
#include "stats/stats_compute.hpp"
#include "compress/multikey_compress.hpp"
#include "utils/transform.hpp"

namespace gamax::groupby {

namespace {

// Helper to compute AVG by dividing SUM by count
struct compute_avg_helper {
    template <typename SumT>
    void operator()(
        gamax::column_view sum_col,
        gamax::column_view count_col,
        gamax::mutable_column_view avg_col,
        cudaStream_t stream) const
    {
        auto sum_ptr = sum_col.data<SumT>();
        auto count_ptr = count_col.data<int64_t>();
        auto avg_ptr = avg_col.data<double>();
        auto n = sum_col.size();
        
        thrust::transform(
            thrust::cuda::par.on(stream),
            sum_ptr, sum_ptr + n,
            count_ptr,
            avg_ptr,
            [] __device__ (SumT sum, int64_t count) -> double {
                return static_cast<double>(sum) / static_cast<double>(count);
            });
    }
};

void compute_avg_from_sum(
    gamax::column_view sum_col,
    gamax::column_view count_col,
    gamax::mutable_column_view avg_col,
    cudaStream_t stream)
{
    gamax::type_dispatcher(
        sum_col.type(),
        compute_avg_helper{},
        sum_col,
        count_col,
        avg_col,
        stream);
}

void auto_select_groupby_algo(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    if(hint.groupby_algo != gamax::groupby::AUTO_ALGO) return;

    // STRING keys → HASH. Only the hash path has STRING-aware
    // row_hasher / row_equal; sort and partition assume trivially
    // comparable numeric keys. This check runs first so `compute_ndv_approx`
    // below (which uses the STRING-unaware type_dispatcher) never sees a
    // STRING column.
    for(gamax::size_type i = 0; i < key.num_columns(); i++) {
        if(key.get_column(i).type().id() == gamax::type_id::STRING) {
            hint.groupby_algo = gamax::groupby::HASH;
            return;
        }
    }

    // Multi-col numeric keys → HASH. If the caller wanted SKF's key
    // fusion, they already ran it in `groupby()` top-level, and by
    // the time we see multi-col here it means either SKF is disabled
    // or compression gave up (bit budget). The cardinality heuristic
    // below requires a single column; HASH is the safe functional
    // default for multi-col — a multi-col partition heuristic would
    // need an NDV estimator we don't have.
    if(key.num_columns() != 1) {
        hint.groupby_algo = gamax::groupby::HASH;
        return;
    }

    auto est_groups = gamax::stats::compute_ndv_approx(
        key.get_column(0),
        stream).get<gamax::size_type>();


    bool is_4_byte_key = (gamax::size_of(key.get_column(0).type()) == 4);

    if(is_4_byte_key) {
        if(est_groups <= (static_cast<gamax::size_type>(1) << 14)) {
            hint.groupby_algo = gamax::groupby::HASH;
        } else {
            hint.groupby_algo = gamax::groupby::PARTITION;
        }
    } else {
        if(est_groups <= (static_cast<gamax::size_type>(1) << 18)) {
            hint.groupby_algo = gamax::groupby::HASH;
        } else {
            hint.groupby_algo = gamax::groupby::PARTITION;
        }
    }
}

std::unique_ptr<gamax::table> groupby_noskf(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{   
    // Track which columns have AVG requests
    std::vector<gamax::size_type> avg_column_indices;
    // Track which columns already have SUM requests (to avoid duplicates)
    std::unordered_set<gamax::size_type> sum_column_indices;
    
    // First pass: identify columns with SUM requests
    for (auto const& req : agg_requests) {
        if (req.agg_kind == gamax::groupby::aggregation_id::SUM) {
            sum_column_indices.insert(req.column_index);
        }
    }
    
    // Second pass: identify columns with AVG requests
    for (auto const& req : agg_requests) {
        if (req.agg_kind == gamax::groupby::aggregation_id::AVG) {
            avg_column_indices.push_back(req.column_index);
        }
    }
    
    bool has_avg = !avg_column_indices.empty();
    bool has_count_all = std::any_of(
        agg_requests.begin(), agg_requests.end(),
        [](auto const& r) {
            return r.agg_kind == gamax::groupby::aggregation_id::COUNT_ALL;
        });
    // The internal "ones" column built for AVG's denominator also
    // serves COUNT_ALL — so adding COUNT_ALL alongside AVG costs
    // nothing extra. Allocate it whenever any of the three
    // count-needers is present.
    bool needs_count = hint.agg_count || has_avg || has_count_all;
    
    // Build internal aggregation requests: replace AVG with SUM, avoid duplicates
    std::vector<gamax::groupby::aggregation_request> internal_agg_requests;
    // Map from original request index to internal result column index
    std::vector<gamax::size_type> result_col_mapping;
    // Track internal SUM columns for each (column_index, SUM) pair
    std::unordered_map<gamax::size_type, gamax::size_type> sum_result_col_index;
    
    gamax::size_type internal_result_idx = 0; // Start after the key column
    for (size_t i = 0; i < agg_requests.size(); ++i) {
        auto const& req = agg_requests[i];

        if (req.agg_kind == gamax::groupby::aggregation_id::AVG) {
            // Check if we already have a SUM for this column
            auto it = sum_result_col_index.find(req.column_index);
            if (it != sum_result_col_index.end()) {
                // Reuse existing SUM result
                result_col_mapping.push_back(it->second);
            } else {
                // Add a SUM request for this column
                internal_agg_requests.push_back(
                    gamax::groupby::aggregation_request(req.column_index, gamax::groupby::aggregation_id::SUM));
                sum_result_col_index[req.column_index] = internal_result_idx;
                result_col_mapping.push_back(internal_result_idx);
                internal_result_idx++;
            }
        } else if (req.agg_kind == gamax::groupby::aggregation_id::COUNT_ALL) {
            // COUNT_ALL maps to the shared internal count column,
            // whose index is only known after the existing
            // (AVG/SUM/MAX) requests are placed. Stash a placeholder
            // and patch it up below once count_col_internal_idx is
            // assigned.
            result_col_mapping.push_back(/*placeholder*/ -1);
        } else {
            // SUM, MAX, etc. - add directly
            internal_agg_requests.push_back(req);
            if (req.agg_kind == gamax::groupby::aggregation_id::SUM) {
                sum_result_col_index[req.column_index] = internal_result_idx;
            }
            result_col_mapping.push_back(internal_result_idx);
            internal_result_idx++;
        }
    }
    
    // Create ones column for counting if needed
    column_ptr ones = nullptr;
    gamax::size_type count_col_internal_idx = -1;
    std::vector<gamax::column_view> extended_val_cols;
    
    if (needs_count) {
        // Copy existing val columns
        for (gamax::size_type i = 0; i < val.num_columns(); ++i) {
            extended_val_cols.push_back(val.get_column(i));
        }

        // Use key.num_rows() to size the ones column. val.num_rows()
        // returns 0 when val has 0 columns (e.g. an Aggregate whose
        // only request is COUNT_ALL — common after the rule layer
        // stops materializing the lit(1) body), which would zero
        // out every count. Keys always carry the input row count.
        const gamax::size_type n_rows = key.num_rows();
        ones = gamax::make_column<int64_t>(n_rows, false, stream);
        // Fill with 1s
        auto ones_ptr = ones->data<int64_t>();
        thrust::fill(thrust::cuda::par.on(stream), ones_ptr,
                     ones_ptr + n_rows, static_cast<int64_t>(1));
        extended_val_cols.push_back(ones->view());
        
        // Add count aggregation request (SUM of ones column)
        count_col_internal_idx = internal_result_idx;
        internal_agg_requests.push_back(
            gamax::groupby::aggregation_request(val.num_columns(), gamax::groupby::aggregation_id::SUM));
        internal_result_idx++;
    }

    // Patch up COUNT_ALL placeholders now that count_col_internal_idx
    // is assigned. Safe to do unconditionally: if no COUNT_ALL items
    // were present, this loop is a no-op.
    if (has_count_all) {
        for (size_t i = 0; i < agg_requests.size(); ++i) {
            if (agg_requests[i].agg_kind ==
                gamax::groupby::aggregation_id::COUNT_ALL)
            {
                result_col_mapping[i] = count_col_internal_idx;
            }
        }
    }
    
    // Create extended val table view
    gamax::table_view extended_val = needs_count ? gamax::table_view(extended_val_cols) : val;
    
    auto_select_groupby_algo(key, hint, stream);

    // Optional fallback: opt-in via hint.allow_hash_fallback. If a non-HASH
    // algo was requested but the input shape (STRING key, or multi-column
    // key) isn't supported by sort-/partition-based paths, transparently
    // downgrade to HASH. This mirrors the predicate that
    // `auto_select_groupby_algo` already applies to AUTO_ALGO callers
    // (STRING/multi-col → HASH), but extends it to explicit-hint callers
    // that opt in. Used by the bench harness to keep multi-step queries
    // like Q21 sweep-able across non-HASH group_by axes — the inner
    // integer-keyed groupbys exercise the configured algo, while the
    // outer STRING-keyed groupby silently falls back to HASH.
    if (hint.allow_hash_fallback &&
        hint.groupby_algo != gamax::groupby::HASH)
    {
        bool has_string_key = false;
        for (gamax::size_type i = 0; i < key.num_columns(); ++i) {
            if (key.get_column(i).type().id() == gamax::type_id::STRING) {
                has_string_key = true;
                break;
            }
        }
        if (has_string_key || key.num_columns() > 1) {
            hint.groupby_algo = gamax::groupby::HASH;
        }
    }

    // PARTITION groupby's underlying impl only handles a single
    // numeric key column — its result schema is `[encoded_key,
    // agg0..]` regardless of caller-provided key cardinality, so the
    // post-processing here would index past the result's columns for
    // multi-key / STRING inputs (surfacing as "column out of range"
    // or `vector::_M_range_check`). Reject those cases up front with
    // a clear error so the bench harness reports a methodologically
    // useful "unsupported shape" instead of a cryptic indexing
    // throw. `auto_select_groupby_algo` already routes AUTO callers
    // away from PARTITION on the same conditions; this is the
    // explicit-hint counterpart of that guard.
    if (hint.groupby_algo == gamax::groupby::PARTITION ||
        hint.groupby_algo == gamax::groupby::PARTITION_GFUR)
    {
        if (key.num_columns() != 1) {
            throw std::runtime_error(
                "PARTITION groupby: only single-column keys supported "
                "(got " + std::to_string(key.num_columns()) + ")");
        }
        if (key.get_column(0).type().id() == gamax::type_id::STRING) {
            throw std::runtime_error(
                "PARTITION groupby: STRING keys not supported");
        }
    }

    // Execute the internal groupby
    std::unique_ptr<gamax::table> internal_result;
    switch (hint.groupby_algo) {
        case gamax::groupby::SORT_OPT:
            hint.optimize_sort = true;
            [[fallthrough]];
        case gamax::groupby::SORT:
            internal_result = gamax::groupby::sort::groupby(key, extended_val, internal_agg_requests, hint, stream);
            break;
        case gamax::groupby::SORT_GFUR:
            internal_result = gamax::groupby::sort::groupby_gfur(key, extended_val, internal_agg_requests, hint, stream);
            break;
        case gamax::groupby::PARTITION:
            internal_result = gamax::groupby::partition::groupby(key, extended_val, internal_agg_requests, hint, stream);
            break;
        case gamax::groupby::PARTITION_GFUR:
            internal_result = gamax::groupby::partition::groupby_gfur(key, extended_val, internal_agg_requests, hint, stream);
            break;
        case gamax::groupby::HASH:
            internal_result = gamax::groupby::hash::groupby(key, extended_val, internal_agg_requests, hint, stream);
            break;
        default:
            throw std::runtime_error("Unsupported groupby algorithm");
    }
    
    // Build final result table
    // Internal result has: [key_cols..., agg_col_0, agg_col_1, ..., count_col (if needed)]
    std::vector<std::unique_ptr<gamax::column>> final_columns;
    
    gamax::size_type num_key_cols = key.num_columns();
    
    // Get count column view if needed (before we start removing columns)
    // Count column is at index: num_key_cols + count_col_internal_idx
    gamax::column_view count_col_view;
    if (has_avg) {
        count_col_view = internal_result->get_column(num_key_cols + count_col_internal_idx).view();
    }
    
    // Move key columns out of internal_result rather than deep_copy
    // them — internal_result is discarded at function return anyway,
    // and `column::deep_copy` uses `size_of(type)` which doesn't
    // handle STRING. `release_column` leaves nullptr in the vector;
    // the only subsequent reads from internal_result are into agg
    // cols (indices ≥ num_key_cols), so the nullptrs we leave here
    // aren't touched.
    for (gamax::size_type k = 0; k < num_key_cols; ++k) {
        final_columns.push_back(internal_result->release_column(k));
    }
    
    gamax::size_type num_groups = final_columns[0]->size();
    
    // Process each original aggregation request
    // Aggregation columns start at index num_key_cols in internal_result
    for (size_t i = 0; i < agg_requests.size(); ++i) {
        auto const& req = agg_requests[i];
        gamax::size_type internal_col_idx = num_key_cols + result_col_mapping[i];
        
        if (req.agg_kind == gamax::groupby::aggregation_id::AVG) {
            // Create AVG column (always double)
            auto avg_col = gamax::make_column<double>(num_groups, false, stream);

            // Get the SUM column for this AVG
            auto sum_col_view = internal_result->get_column(internal_col_idx).view();

            // Compute AVG = SUM / count
            compute_avg_from_sum(sum_col_view, count_col_view, avg_col->mutable_view(), stream);

            final_columns.push_back(std::move(avg_col));
        } else if (req.agg_kind == gamax::groupby::aggregation_id::COUNT_ALL) {
            // COUNT_ALL items share the internal count column
            // (already populated as SUM of the ones-broadcast). Copy
            // it out so each COUNT_ALL output is independently
            // owned.
            final_columns.push_back(
                internal_result->get_column(
                    num_key_cols + count_col_internal_idx).deep_copy(stream));
        } else {
            // For SUM, MAX, etc., copy the column directly
            // TODO: check again, why copying?
            final_columns.push_back(internal_result->get_column(internal_col_idx).deep_copy(stream));
        }
    }
    
    // Add count column if requested by hint
    // Count column is at index: num_key_cols + count_col_internal_idx
    if (hint.agg_count) {
        final_columns.push_back(internal_result->get_column(num_key_cols + count_col_internal_idx).deep_copy(stream));
    }
    
    cudaStreamSynchronize(stream);
    
    return std::make_unique<gamax::table>(std::move(final_columns));
}
} // anonymous namespace

std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    if (!hint.enable_skf) {
        return groupby_noskf(key, val, agg_requests, hint, stream);
    }

    int compressed_bits = 0;
    gamax::compress::multikey_compress_options compress_options;
    compress_options.out_total_bits = &compressed_bits;

    auto compressed_key = gamax::compress::multikey_compress(key, compress_options, stream);
    if(!compressed_key) {
        return groupby_noskf(key, val, agg_requests, hint, stream);
    }

    hint.groups_upper_bound = std::min(hint.groups_upper_bound, static_cast<gamax::size_type>(1) << compressed_bits);

    std::vector<column_ptr> compressed_key_vec;
    compressed_key_vec.push_back(std::move(compressed_key));
    auto combined_key_table = gamax::make_table(std::move(compressed_key_vec));

    auto tids = gamax::make_sequence<gamax::size_type>(key.num_rows(), 0, 1, stream);
    // `table_view::add_column` requires the new column's size to match
    // `num_rows()`, but `num_rows()` returns 0 when val has 0 columns
    // (e.g. an Aggregate whose only request is COUNT_ALL — no body
    // materialized). Reconstruct val with tids prepended in that case,
    // and append in the normal case.
    gamax::table_view val_with_tids;
    if (val.num_columns() == 0) {
        std::vector<gamax::column_view> cols{tids->view()};
        val_with_tids = gamax::table_view(cols);
    } else {
        val_with_tids = val;
        val_with_tids.add_column(tids->view());
    }

    // Make a local copy to avoid mutating caller's agg_requests
    auto local_agg_requests = agg_requests;
    local_agg_requests.push_back(gamax::groupby::aggregation_request(val_with_tids.num_columns() - 1, gamax::groupby::aggregation_id::ANY));
    
    auto result = groupby_noskf(combined_key_table->view(), val_with_tids, local_agg_requests, hint, stream);

    // result schema: [compressed_key, agg_col_0, agg_col_1, ..., ANY(tids), count (if agg_count == true)]
    auto tid_idx = result->num_columns() - 1;
    if(hint.agg_count) {
        tid_idx -= 1; // if count is needed, the ANY column is before the count column
    }

    // Remove the ANY column (tids)
    auto res_tids = result->remove_column(tid_idx);

    // Gather the original key columns using the tids
    std::vector<column_ptr> final_columns;
    for(int i = 0; i < key.num_columns(); i++) {
        auto res_key = gamax::util::device::gather(res_tids->view(), key.get_column(i), stream);
        final_columns.push_back(std::move(res_key));
    }
    for(int i = 1; i < result->num_columns(); i++) {
        final_columns.push_back(std::move(result->release_column(i)));
    }

    cudaStreamSynchronize(stream);

    return gamax::make_table(std::move(final_columns));
}
} // namespace gamax::groupby
