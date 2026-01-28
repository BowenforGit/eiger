#include "groupby.hpp"

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
    if(hint.groupby_algo != gamax::groupby::AUTO_ALGO || key.num_columns() != 1) {
        // Currently, only auto-select for single-column keys
        return;
    }

    auto est_groups = gamax::stats::compute_ndv_approx(
        key.get_column(0),
        stream).get<gamax::size_type>();
    
    est_groups = std::min(est_groups, hint.groups_upper_bound);
    
    if(est_groups < (static_cast<gamax::size_type>(1) << 20)) {
        hint.groupby_algo = gamax::groupby::HASH;
    } else {
        hint.groupby_algo = gamax::groupby::SORT;
    }
}

} // anonymous namespace

std::unique_ptr<gamax::table> groupby(
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
    bool needs_count = hint.agg_count || has_avg;
    
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
        
        // Create ones column and add to extended val
        ones = gamax::make_column<int64_t>(val.num_rows(), false, stream);
        // Fill with 1s
        auto ones_ptr = ones->data<int64_t>();
        thrust::fill(thrust::cuda::par.on(stream), ones_ptr, ones_ptr + val.num_rows(), static_cast<int64_t>(1));
        extended_val_cols.push_back(ones->view());
        
        // Add count aggregation request (SUM of ones column)
        count_col_internal_idx = internal_result_idx;
        internal_agg_requests.push_back(
            gamax::groupby::aggregation_request(val.num_columns(), gamax::groupby::aggregation_id::SUM));
        internal_result_idx++;
    }
    
    // Create extended val table view
    gamax::table_view extended_val = needs_count ? gamax::table_view(extended_val_cols) : val;
    
    // Execute the internal groupby
    std::unique_ptr<gamax::table> internal_result;
    switch (hint.groupby_algo) {
        case gamax::groupby::AUTO_ALGO:
        case gamax::groupby::SORT:
            internal_result = gamax::groupby::sort::groupby(key, extended_val, internal_agg_requests, hint, stream);
            break;
        case gamax::groupby::SORT_GFUR:
            internal_result = gamax::groupby::sort::groupby_gfur(key, extended_val, internal_agg_requests, hint, stream);
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
    
    // Add all key columns from internal result
    for (gamax::size_type k = 0; k < num_key_cols; ++k) {
        final_columns.push_back(internal_result->get_column(k).deep_copy(stream));
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
        } else {
            // For SUM, MAX, etc., copy the column directly
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
} // namespace gamax::groupby