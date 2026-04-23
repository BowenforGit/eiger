#include <memory>
#include <utility>
#include <vector>
#include <stdexcept>
#include <limits>

#include <cuda.h>
#include <cub/cub.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "operators/sort/sort.hpp"
#include "operators/sort/sort_helpers.cuh"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "string/string_impl.cuh"
#include "utils/cub_wrapper.cuh"
#include "utils/transform.hpp"

namespace gamax::sort {
namespace {

    // =========================================================================
    // Helper: Check if we can use int32 for row IDs (more efficient sorting)
    // =========================================================================
    __host__ __forceinline__
    constexpr bool can_use_int32_rid(gamax::size_type num_rows) {
        return num_rows <= static_cast<gamax::size_type>(std::numeric_limits<int32_t>::max());
    }

    // =========================================================================
    // Helper: Apply sorted indices to reorder non-key columns
    // =========================================================================
    void apply_sorted_indices_to_columns(
        gamax::table_view const& table,
        gamax::column_view const& sorted_indices,
        std::vector<gamax::size_type> const& numeric_col_indices,
        std::vector<gamax::size_type> const& string_col_indices,
        std::vector<std::unique_ptr<gamax::column>>& sorted_columns,
        cudaStream_t stream)
    {
        // For numeric columns: sort with the sorted indices
        for (auto col_idx : numeric_col_indices) {
            auto [_, sorted_col] = sort_column_pair(
                sorted_indices,
                table.get_column(col_idx),
                0, -1, gamax::sort_order::ASCENDING, nullptr, 0, stream);
            sorted_columns[col_idx] = std::move(sorted_col);
        }
        
        // For string columns: gather using sorted indices
        for (auto col_idx : string_col_indices) {
            auto col_view = table.get_column(col_idx);
            gamax::string_column_view str_view(col_view, gamax::ARROW_STRING, stream);
            sorted_columns[col_idx] = gamax::gather_strings(str_view, sorted_indices, stream);
        }
    }

    // =========================================================================
    // Single-key sorting: Numeric key column
    // =========================================================================
    std::unique_ptr<gamax::table>
    sort_numeric_column_table(
        gamax::table_view const& table,
        int sort_key_index,
        gamax::sort_order sort_order,
        cudaStream_t stream)
    {
        const auto num_cols = table.num_columns();
        const auto num_rows = table.num_rows();
        
        if (num_rows == 0) {
            return detail::make_empty_table_like(table, stream);
        }

        auto sort_key_col = table.get_column(sort_key_index);
        
        // Classify columns by type
        std::vector<gamax::size_type> numeric_col_indices;
        std::vector<gamax::size_type> string_col_indices;
        
        for (gamax::size_type i = 0; i < num_cols; ++i) {
            if (i == static_cast<gamax::size_type>(sort_key_index)) continue;
            if (table.get_column(i).type().id() == gamax::type_id::STRING) {
                string_col_indices.push_back(i);
            } else {
                numeric_col_indices.push_back(i);
            }
        }
        
        std::vector<std::unique_ptr<gamax::column>> sorted_columns(num_cols);
        
        // For numeric columns (including the key), use sort_column_pair
        bool key_sorted = false;
        
        if (!numeric_col_indices.empty()) {
            // Sort key with first numeric column
            auto [sorted_key, sorted_val] = sort_column_pair(
                sort_key_col,
                table.get_column(numeric_col_indices[0]),
                0, -1, sort_order, nullptr, 0, stream);
            
            sorted_columns[sort_key_index] = std::move(sorted_key);
            sorted_columns[numeric_col_indices[0]] = std::move(sorted_val);
            key_sorted = true;
            
            // Sort remaining numeric columns with the key
            for (size_t i = 1; i < numeric_col_indices.size(); ++i) {
                auto [_, sorted_col] = sort_column_pair(
                    sort_key_col,
                    table.get_column(numeric_col_indices[i]),
                    0, -1, sort_order, nullptr, 0, stream);
                sorted_columns[numeric_col_indices[i]] = std::move(sorted_col);
            }
        }
        
        // Handle string columns - need sorted indices (argsort) only once
        if (!string_col_indices.empty()) {
            auto [sorted_key, sorted_indices] = argsort(sort_key_col, 0, -1, sort_order, nullptr, 0, stream);
            
            if (!key_sorted) {
                sorted_columns[sort_key_index] = std::move(sorted_key);
                key_sorted = true;
            }
            
            for (auto col_idx : string_col_indices) {
                auto col_view = table.get_column(col_idx);
                gamax::string_column_view str_view(col_view, gamax::ARROW_STRING, stream);
                sorted_columns[col_idx] = gamax::gather_strings(str_view, sorted_indices->view(), stream);
            }
        }
        
        // If only the key column exists and it's numeric
        if (!key_sorted) {
            sorted_columns[sort_key_index] = sort_column(sort_key_col, 0, -1, sort_order, nullptr, 0, stream);
        }
        
        return gamax::make_table(std::move(sorted_columns));
    }

    // =========================================================================
    // Single-key sorting: String key column
    // =========================================================================
    template <typename RidType>
    std::unique_ptr<gamax::table>
    sort_string_column_table_impl(
        gamax::table_view const& table,
        int sort_key_index,
        gamax::sort_order sort_order,
        cudaStream_t stream)
    {
        const auto num_cols = table.num_columns();
        const auto num_rows = table.num_rows();
        
        if (num_rows == 0) {
            return detail::make_empty_table_like(table, stream);
        }
        
        auto sort_key_col_view = table.get_column(sort_key_index);
        gamax::string_column_view sort_key_str_view(sort_key_col_view, gamax::ARROW_STRING, stream);

        // Step 1: argsort_strings to get sorted indices A (returns size_type)
        auto sorted_indices_A = gamax::argsort_strings(sort_key_str_view, sort_order, stream);

        // Step 2: Create index column B (0 to N-1) using RidType and sort with A
        auto indices_B = gamax::make_sequence<RidType>(num_rows, 0, 1, stream);
        auto [sorted_A, sorted_B] = sort_column_pair(
            sorted_indices_A->view(), indices_B->view(), 0, -1,
            gamax::sort_order::ASCENDING, nullptr, 0, stream);
        
        // Classify columns
        std::vector<gamax::size_type> numeric_col_indices;
        std::vector<gamax::size_type> string_col_indices;
        
        for (gamax::size_type i = 0; i < num_cols; ++i) {
            if (table.get_column(i).type().id() == gamax::type_id::STRING) {
                string_col_indices.push_back(i);
            } else {
                numeric_col_indices.push_back(i);
            }
        }
        
        std::vector<std::unique_ptr<gamax::column>> sorted_columns(num_cols);

        // Numeric columns go through sort_column_pair using the
        // RidType-typed sorted_B. String columns are handled separately
        // below with sorted_indices_A (size_type), since `gather_strings`
        // expects a size_type gather map — passing an int32 RidType-map
        // would read past the buffer end.
        std::vector<gamax::size_type> empty_string_indices;
        apply_sorted_indices_to_columns(
            table, sorted_B->view(), numeric_col_indices, empty_string_indices,
            sorted_columns, stream);

        // Gather the string columns using the size_type sorted_indices_A.
        for (auto col_idx : string_col_indices) {
            auto col_view = table.get_column(col_idx);
            gamax::string_column_view str_view(col_view, gamax::ARROW_STRING, stream);
            sorted_columns[col_idx] = gamax::gather_strings(str_view, sorted_indices_A->view(), stream);
        }
        
        return gamax::make_table(std::move(sorted_columns));
    }

    // Dispatch function that chooses int32 or size_type based on table size
    std::unique_ptr<gamax::table>
    sort_string_column_table(
        gamax::table_view const& table,
        int sort_key_index,
        gamax::sort_order sort_order,
        cudaStream_t stream)
    {
        if (can_use_int32_rid(table.num_rows())) {
            return sort_string_column_table_impl<int32_t>(table, sort_key_index, sort_order, stream);
        } else {
            return sort_string_column_table_impl<gamax::size_type>(table, sort_key_index, sort_order, stream);
        }
    }

    // =========================================================================
    // Multi-key sorting: Row comparator
    // =========================================================================
    
    // Maximum number of sort keys supported (for fixed-size device arrays)
    constexpr int MAX_SORT_KEYS = 16;
    
    // Device function to compare two strings
    // Note: We use gamax::size_type for indexing offsets array since offsets are always size_type
    template <typename RidType>
    __device__ __forceinline__
    int compare_strings(
        const char* data, const gamax::size_type* offsets,
        RidType lhs_idx, RidType rhs_idx)
    {
        gamax::size_type lhs_start = offsets[lhs_idx];
        gamax::size_type lhs_end = offsets[lhs_idx + 1];
        gamax::size_type rhs_start = offsets[rhs_idx];
        gamax::size_type rhs_end = offsets[rhs_idx + 1];
        
        gamax::size_type lhs_len = lhs_end - lhs_start;
        gamax::size_type rhs_len = rhs_end - rhs_start;
        gamax::size_type min_len = lhs_len < rhs_len ? lhs_len : rhs_len;
        
        const char* lhs_data = data + lhs_start;
        const char* rhs_data = data + rhs_start;
        
        for (gamax::size_type i = 0; i < min_len; ++i) {
            if (lhs_data[i] < rhs_data[i]) return -1;
            if (lhs_data[i] > rhs_data[i]) return 1;
        }
        
        if (lhs_len < rhs_len) return -1;
        if (lhs_len > rhs_len) return 1;
        return 0;
    }
    
    // Functor to compare two elements of a column by their indices
    // Uses type_dispatcher_all for automatic type dispatching including STRING
    // RidType is the row ID type (int32 or size_type)
    template <typename RidType>
    struct CompareElementsFunctor {
        template<typename T>
        __device__ __forceinline__
        int operator()(gamax::column_view const& col, RidType lhs_idx, RidType rhs_idx) const
        {
            if constexpr (cuda::std::is_same_v<T, gamax::string_type_tag>) {
                // String comparison using ARROW format
                // For string columns, data pointer points to a helper buffer containing [char_data_ptr, offsets_ptr]
                // Must use reinterpret_cast to access the pointers correctly (see string_column_view device constructor)
                void** ptrs = reinterpret_cast<void**>(const_cast<void*>(col.data<void>()));
                const char* char_data = static_cast<const char*>(ptrs[0]);
                const gamax::size_type* offsets = static_cast<const gamax::size_type*>(ptrs[1]);
                return compare_strings<RidType>(char_data, offsets, lhs_idx, rhs_idx);
            } else {
                // Numeric comparison - array indexing works with any integer type
                const T* typed_data = col.data<T>();
                T lhs = typed_data[lhs_idx];
                T rhs = typed_data[rhs_idx];
                if (lhs < rhs) return -1;
                if (lhs > rhs) return 1;
                return 0;
            }
        }
    };
    
    // Multi-key row comparator using table_device_view
    // RidType is the row ID type used for indices (int32 for smaller tables, size_type for large)
    template <typename RidType>
    struct MultiKeyRowComparator {
        gamax::experimental::table_device_view keys_view;
        bool sort_orders[MAX_SORT_KEYS];  // true = ascending
        int num_keys;
        
        __host__ __device__
        MultiKeyRowComparator(
            gamax::experimental::table_device_view keys,
            const bool* orders,
            int n_keys)
            : keys_view(keys), num_keys(n_keys)
        {
            for (int i = 0; i < n_keys && i < MAX_SORT_KEYS; ++i) {
                sort_orders[i] = orders[i];
            }
        }
        
        __device__
        bool operator()(RidType lhs_idx, RidType rhs_idx) const
        {
            for (int k = 0; k < num_keys; ++k) {
                auto const& col = keys_view.get_column(k);
                int cmp = gamax::type_dispatcher_all(
                    col.type().id(), 
                    CompareElementsFunctor<RidType>{},
                    col,
                    lhs_idx,
                    rhs_idx
                );
                
                if (cmp != 0) {
                    // Apply sort order: ascending means lhs < rhs returns true
                    // descending means lhs > rhs returns true (flip the comparison)
                    return sort_orders[k] ? (cmp < 0) : (cmp > 0);
                }
            }
            
            // All keys equal - stable sort: preserve original order
            return lhs_idx < rhs_idx;
        }
    };

    // =========================================================================
    // Multi-key sorting implementation
    // =========================================================================
    template <typename RidType>
    std::unique_ptr<gamax::table>
    sort_multi_key_table_impl(
        gamax::table_view const& table,
        std::vector<int> const& sort_keys,
        std::vector<gamax::sort_order> const& sort_orders,
        cudaStream_t stream)
    {
        const auto num_cols = table.num_columns();
        const auto num_rows = table.num_rows();
        const auto num_keys = static_cast<int>(sort_keys.size());
        
        if (num_rows == 0) {
            return detail::make_empty_table_like(table, stream);
        }
        
        if (num_keys > MAX_SORT_KEYS) {
            throw std::invalid_argument("Too many sort keys (max " + std::to_string(MAX_SORT_KEYS) + ")");
        }
        
        // Build key columns table view and create device view
        std::vector<gamax::size_type> key_indices(num_keys);
        for (int k = 0; k < num_keys; ++k) {
            key_indices[k] = sort_keys[k];
        }
        auto keys_table_view = table.select_columns(key_indices);
        auto keys_device_view = keys_table_view.device_view(stream);
        
        // Build sort orders array on host
        bool host_sort_orders[MAX_SORT_KEYS];
        for (int k = 0; k < num_keys; ++k) {
            host_sort_orders[k] = (sort_orders[k] == gamax::sort_order::ASCENDING);
        }
        
        // Create comparator with device view and sort orders
        MultiKeyRowComparator<RidType> comparator(*keys_device_view, host_sort_orders, num_keys);
        
        // Initialize indices (0 to N-1) using RidType for efficiency
        auto indices = gamax::make_sequence<RidType>(num_rows, 0, 1, stream);
        
        // Sort indices using multi-key comparator
        CALL_CUB_FUNC(
            cub::DeviceMergeSort::SortKeys,
            indices->template data<RidType>(),
            num_rows,
            comparator,
            stream);
        
        cudaStreamSynchronize(stream);
        
        // Now we have sorted indices. Apply them to all columns.
        // Create inverse permutation for efficient numeric column sorting using RidType
        // auto indices_B = gamax::make_sequence<RidType>(num_rows, 0, 1, stream);
        // auto [sorted_indices, inverse_perm] = sort_column_pair(
        //     indices->view(), indices_B->view(), 0, -1,
        //     gamax::sort_order::ASCENDING, nullptr, 0, stream);
        
        // Classify all columns
        std::vector<gamax::size_type> numeric_col_indices;
        std::vector<gamax::size_type> string_col_indices;
        
        for (gamax::size_type i = 0; i < num_cols; ++i) {
            if (table.get_column(i).type().id() == gamax::type_id::STRING) {
                string_col_indices.push_back(i);
            } else {
                numeric_col_indices.push_back(i);
            }
        }
        
        std::vector<std::unique_ptr<gamax::column>> sorted_columns(num_cols);
        
        // For numeric columns: use inverse_perm with sort_column_pair
        for (auto col_idx : numeric_col_indices) {
            // auto [_, sorted_col] = sort_column_pair(
            //     inverse_perm->view(),
            //     table.get_column(col_idx),
            //     0, -1, gamax::sort_order::ASCENDING, nullptr, 0, stream);
            auto sorted_col = gamax::util::device::gather(
                indices->view(), table.get_column(col_idx), stream
            );
            sorted_columns[col_idx] = std::move(sorted_col);
        }
        
        // For string columns: gather using original sorted indices
        // Note: gather_strings expects size_type indices, so we need to convert if RidType is different
        if (!string_col_indices.empty()) {
            gamax::column_view indices_for_gather = indices->view();
            std::unique_ptr<gamax::column> size_type_indices;
            
            if constexpr (!std::is_same_v<RidType, gamax::size_type>) {
                // Convert indices from RidType to size_type for gather_strings
                size_type_indices = gamax::make_column<gamax::size_type>(num_rows, false, stream);
                auto* src = indices->template data<RidType>();
                auto* dst = size_type_indices->template data<gamax::size_type>();
                thrust::transform(
                    thrust::cuda::par.on(stream),
                    src, src + num_rows,
                    dst,
                    [] __device__ (RidType val) { return static_cast<gamax::size_type>(val); }
                );
                indices_for_gather = size_type_indices->view();
            }
            
            for (auto col_idx : string_col_indices) {
                auto col_view = table.get_column(col_idx);
                gamax::string_column_view str_view(col_view, gamax::ARROW_STRING, stream);
                sorted_columns[col_idx] = gamax::gather_strings(str_view, indices_for_gather, stream);
            }
        }
        
        return gamax::make_table(std::move(sorted_columns));
    }

    // Dispatch function that chooses int32 or size_type based on table size
    std::unique_ptr<gamax::table>
    sort_multi_key_table(
        gamax::table_view const& table,
        std::vector<int> const& sort_keys,
        std::vector<gamax::sort_order> const& sort_orders,
        cudaStream_t stream)
    {
        if (can_use_int32_rid(table.num_rows())) {
            return sort_multi_key_table_impl<int32_t>(table, sort_keys, sort_orders, stream);
        } else {
            return sort_multi_key_table_impl<gamax::size_type>(table, sort_keys, sort_orders, stream);
        }
    }

} // namespace

    std::unique_ptr<gamax::table>
    sort_table(
        gamax::table_view const& table,
        std::vector<int> const& sort_keys,
        std::vector<gamax::sort_order> const& sort_orders,
        cudaStream_t stream)
    {
        if(sort_keys.empty()) {
            throw std::invalid_argument("sort_keys cannot be empty");
        }

        if (sort_keys.size() != sort_orders.size()) {
            throw std::invalid_argument("sort_keys and sort_orders must have the same size");
        }

        if(sort_keys.size() == 1) {
            return sort_table(table, sort_keys[0], sort_orders[0], stream);
        }

        return sort_multi_key_table(table, sort_keys, sort_orders, stream);
    }

    std::unique_ptr<gamax::table>
    sort_table(
        gamax::table_view const& table,
        int sort_key_index,
        gamax::sort_order sort_order,
        cudaStream_t stream)
    {
        auto sort_key_col = table.get_column(sort_key_index);
        if(sort_key_col.type().id() == gamax::type_id::STRING) {
            return sort_string_column_table(table, sort_key_index, sort_order, stream);
        } else {
            return sort_numeric_column_table(table, sort_key_index, sort_order, stream);
        }
    }
} // namespace gamax::sort