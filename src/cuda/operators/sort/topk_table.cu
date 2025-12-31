#include <memory>
#include <utility>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include <cuda.h>
#include <cuda/std/execution>
#include <cub/cub.cuh>
#include <cub/device/device_topk.cuh>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "operators/sort/sort.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "utils/cub_wrapper.cuh"
#include "utils/transform.hpp"

namespace gamax::sort {
namespace {

    // =========================================================================
    // Helper: Create table with top K rows from sorted table
    // =========================================================================
    std::unique_ptr<gamax::table>
    take_first_k_rows(
        gamax::table_view const& sorted_table,
        gamax::size_type k,
        cudaStream_t stream)
    {
        const auto num_cols = sorted_table.num_columns();
        const auto num_rows = sorted_table.num_rows();
        const auto actual_k = std::min(k, num_rows);
        
        std::vector<std::unique_ptr<gamax::column>> result_columns;
        result_columns.reserve(num_cols);
        
        for (gamax::size_type i = 0; i < num_cols; ++i) {
            auto col_view = sorted_table.get_column(i);
            
            if (col_view.type().id() == gamax::type_id::STRING) {
                // For string columns, use gather with indices 0 to k-1
                auto indices = gamax::make_sequence<gamax::size_type>(actual_k, 0, 1, stream);
                gamax::string_column_view str_view(col_view, gamax::ARROW_STRING, stream);
                result_columns.push_back(gamax::gather_strings(str_view, indices->view(), stream));
            } else {
                // For numeric columns, copy first k elements
                auto result_col = gamax::make_column(col_view.type(), actual_k, false, stream);
                cudaMemcpyAsync(
                    result_col->template data(),
                    col_view.data(),
                    actual_k * gamax::size_of(col_view.type()),
                    cudaMemcpyDeviceToDevice,
                    stream);
                result_columns.push_back(std::move(result_col));
            }
        }
        
        return gamax::make_table(std::move(result_columns));
    }

    // =========================================================================
    // Helper: Apply gather map to create result table
    // =========================================================================
    std::unique_ptr<gamax::table>
    gather_table_by_indices(
        gamax::table_view const& table,
        gamax::column_view const& indices,
        cudaStream_t stream)
    {
        const auto num_cols = table.num_columns();
        
        std::vector<std::unique_ptr<gamax::column>> result_columns;
        result_columns.reserve(num_cols);
        
        for (gamax::size_type i = 0; i < num_cols; ++i) {
            auto col_view = table.get_column(i);
            
            if (col_view.type().id() == gamax::type_id::STRING) {
                gamax::string_column_view str_view(col_view, gamax::ARROW_STRING, stream);
                result_columns.push_back(gamax::gather_strings(str_view, indices, stream));
            } else {
                result_columns.push_back(gamax::util::device::gather(indices, col_view, stream));
            }
        }
        
        return gamax::make_table(std::move(result_columns));
    }

    // =========================================================================
    // Check if a type_id is a primitive numeric type (not STRING)
    // =========================================================================
    bool is_primitive_type(gamax::type_id id)
    {
        switch (id) {
            case gamax::type_id::INT8:
            case gamax::type_id::INT16:
            case gamax::type_id::INT32:
            case gamax::type_id::INT64:
            case gamax::type_id::UINT64:
            case gamax::type_id::FLOAT32:
            case gamax::type_id::FLOAT64:
            case gamax::type_id::BOOL8:
                return true;
            default:
                return false;
        }
    }

    // =========================================================================
    // TopK implementation using cub::DeviceTopK for single primitive key
    // =========================================================================
    
    // Functor for dispatching TopK by key type
    struct TopKDispatcher {
        template <typename KeyT>
        std::unique_ptr<gamax::table> operator()(
            gamax::table_view const& table,
            int sort_key_index,
            gamax::sort_order order,
            gamax::size_type k,
            bool result_in_sorted_order,
            cudaStream_t stream) const
        {
            const auto num_rows = table.num_rows();
            const auto actual_k = std::min(k, num_rows);
            
            if (actual_k == 0) {
                // Return empty table with same schema
                std::vector<std::unique_ptr<gamax::column>> empty_columns;
                for (gamax::size_type i = 0; i < table.num_columns(); ++i) {
                    if (table.get_column(i).type().id() == gamax::type_id::STRING) {
                        empty_columns.push_back(std::make_unique<gamax::string_column>());
                    } else {
                        empty_columns.push_back(gamax::make_column(table.get_column(i).type(), 0, false, stream));
                    }
                }
                return gamax::make_table(std::move(empty_columns));
            }
            
            auto sort_key_col = table.get_column(sort_key_index);
            const KeyT* d_keys_in = sort_key_col.data<KeyT>();
            
            // Create output key column and indices (values)
            auto keys_out = gamax::make_column<KeyT>(actual_k, false, stream);
            auto indices_out = gamax::make_column<gamax::size_type>(actual_k, false, stream);
            
            // Create input indices (0 to N-1)
            auto indices_in = gamax::make_sequence<gamax::size_type>(num_rows, 0, 1, stream);
            
            // Create execution environment
            cuda::stream_ref stream_ref{stream};
            auto requirements = cuda::execution::require(
                cuda::execution::determinism::not_guaranteed,
                cuda::execution::output_ordering::unsorted);
            auto env = cuda::std::execution::env{stream_ref, requirements};
            
            // Call appropriate TopK function based on sort order
            if (order == gamax::sort_order::DESCENDING) {
                // Use MaxPairs for descending (largest K)
                CALL_CUB_FUNC(
                    cub::DeviceTopK::MaxPairs,
                    d_keys_in,
                    keys_out->template data<KeyT>(),
                    indices_in->template data<gamax::size_type>(),
                    indices_out->template data<gamax::size_type>(),
                    num_rows,
                    actual_k,
                    env);
            } else {
                // Use MinPairs for ascending (smallest K)
                CALL_CUB_FUNC(
                    cub::DeviceTopK::MinPairs,
                    d_keys_in,
                    keys_out->template data<KeyT>(),
                    indices_in->template data<gamax::size_type>(),
                    indices_out->template data<gamax::size_type>(),
                    num_rows,
                    actual_k,
                    env);
            }
            
            cudaStreamSynchronize(stream);
            
            if (result_in_sorted_order) {
                // Sort the TopK results by key
                auto [sorted_keys, sorted_indices] = sort_column_pair(
                    keys_out->view(),
                    indices_out->view(),
                    0, -1,
                    order,
                    nullptr, 0,
                    stream);
                
                // Gather all columns using sorted indices
                return gather_table_by_indices(table, sorted_indices->view(), stream);
            } else {
                // Return unsorted TopK results
                return gather_table_by_indices(table, indices_out->view(), stream);
            }
        }
    };

    // =========================================================================
    // TopK for single primitive key using cub::DeviceTopK
    // =========================================================================
    std::unique_ptr<gamax::table>
    topk_single_primitive_key(
        gamax::table_view const& table,
        int sort_key_index,
        gamax::sort_order order,
        gamax::size_type k,
        bool result_in_sorted_order,
        cudaStream_t stream)
    {
        auto sort_key_col = table.get_column(sort_key_index);
        return gamax::type_dispatcher(
            sort_key_col.type(),
            TopKDispatcher{},
            table,
            sort_key_index,
            order,
            k,
            result_in_sorted_order,
            stream);
    }

    // =========================================================================
    // Fallback: Sort entire table and take top K
    // =========================================================================
    std::unique_ptr<gamax::table>
    topk_via_full_sort(
        gamax::table_view const& table,
        std::vector<int> const& sort_keys,
        std::vector<gamax::sort_order> const& sort_orders,
        gamax::size_type k,
        bool result_in_sorted_order,
        cudaStream_t stream)
    {
        // Sort the entire table
        auto sorted_table = sort_table(table, sort_keys, sort_orders, stream);
        
        // Take the first K rows
        auto result = take_first_k_rows(sorted_table->view(), k, stream);
        
        // If result should not be in sorted order, we could shuffle it,
        // but typically TopK results are expected in order, so we just return as-is
        // The sorted order is maintained from the full sort
        return result;
    }

} // namespace

    std::unique_ptr<gamax::table>
    topk_table(
        gamax::table_view const& table,
        std::vector<int> const& sort_keys,
        std::vector<gamax::sort_order> const& sort_orders,
        gamax::size_type k,
        bool result_in_sorted_order,
        cudaStream_t stream)
    {
        if (sort_keys.empty()) {
            throw std::invalid_argument("sort_keys cannot be empty");
        }
        
        if (sort_keys.size() != sort_orders.size()) {
            throw std::invalid_argument("sort_keys and sort_orders must have the same size");
        }
        
        if (k <= 0) {
            throw std::invalid_argument("k must be positive");
        }
        
        const auto num_rows = table.num_rows();
        
        // If k >= num_rows, just sort the entire table
        if (k >= num_rows) {
            return sort_table(table, sort_keys, sort_orders, stream);
        }
        
        // Check if we can use the optimized single-key primitive TopK
        if (sort_keys.size() == 1) {
            auto sort_key_col = table.get_column(sort_keys[0]);
            if (is_primitive_type(sort_key_col.type().id())) {
                return topk_single_primitive_key(
                    table,
                    sort_keys[0],
                    sort_orders[0],
                    k,
                    result_in_sorted_order,
                    stream);
            }
        }
        
        // Fallback: sort entire table and take top K
        return topk_via_full_sort(table, sort_keys, sort_orders, k, result_in_sorted_order, stream);
    }

} // namespace gamax::sort
