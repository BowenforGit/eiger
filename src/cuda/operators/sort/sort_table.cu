#include <memory>
#include <utility>
#include <vector>
#include <stdexcept>

#include <cuda.h>
#include <cub/cub.cuh>
#include <cub/device/device_merge_sort.cuh>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "operators/sort/sort.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "string/string_impl.cuh"
#include "utils/cub_wrapper.cuh"

namespace gamax::sort {
namespace {

    // =========================================================================
    // Helper: Create empty table with same schema
    // =========================================================================
    std::unique_ptr<gamax::table>
    make_empty_table_like(gamax::table_view const& table, cudaStream_t stream)
    {
        std::vector<std::unique_ptr<gamax::column>> empty_columns;
        const auto num_cols = table.num_columns();
        for (gamax::size_type i = 0; i < num_cols; ++i) {
            if (table.get_column(i).type().id() == gamax::type_id::STRING) {
                empty_columns.push_back(std::make_unique<gamax::string_column>());
            } else {
                empty_columns.push_back(gamax::make_column(table.get_column(i).type(), 0, false, stream));
            }
        }
        return gamax::make_table(std::move(empty_columns));
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
            return make_empty_table_like(table, stream);
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
    std::unique_ptr<gamax::table>
    sort_string_column_table(
        gamax::table_view const& table,
        int sort_key_index,
        gamax::sort_order sort_order,
        cudaStream_t stream)
    {
        const auto num_cols = table.num_columns();
        const auto num_rows = table.num_rows();
        
        if (num_rows == 0) {
            return make_empty_table_like(table, stream);
        }
        
        auto sort_key_col_view = table.get_column(sort_key_index);
        gamax::string_column_view sort_key_str_view(sort_key_col_view, gamax::ARROW_STRING, stream);
        
        // Step 1: argsort_strings to get sorted indices A
        auto sorted_indices_A = gamax::argsort_strings(sort_key_str_view, sort_order, stream);
        
        // Step 2: Create index column B (0 to N-1) and sort with A
        auto indices_B = gamax::make_sequence<gamax::size_type>(num_rows, 0, 1, stream);
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
        
        // Use sorted_B for numeric columns, sorted_indices_A for string columns
        apply_sorted_indices_to_columns(
            table, sorted_B->view(), numeric_col_indices, string_col_indices,
            sorted_columns, stream);
        
        // Overwrite the string columns with gathering using original sorted_indices_A
        for (auto col_idx : string_col_indices) {
            auto col_view = table.get_column(col_idx);
            gamax::string_column_view str_view(col_view, gamax::ARROW_STRING, stream);
            sorted_columns[col_idx] = gamax::gather_strings(str_view, sorted_indices_A->view(), stream);
        }
        
        return gamax::make_table(std::move(sorted_columns));
    }

    // =========================================================================
    // Multi-key sorting: Row comparator
    // =========================================================================
    
    // Maximum number of sort keys supported (for fixed-size device arrays)
    constexpr int MAX_SORT_KEYS = 16;
    
    // Device-side key column info
    struct KeyColumnInfo {
        gamax::type_id type;
        const void* data;           // For numeric columns: pointer to data
        const void* meta;           // For string columns: pointer to offsets
        bool is_ascending;
    };
    
    // String implementation for ARROW format
    using arrow_string_impl = gamax::detail::id_to_string_impl<gamax::ARROW_STRING>::string_col_impl;
    
    // Device function to compare two strings
    __device__ __forceinline__
    int compare_strings(
        const char* data, const gamax::size_type* offsets,
        gamax::size_type lhs_idx, gamax::size_type rhs_idx)
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
    struct CompareElementsFunctor {
        const void* data;
        const void* meta;  // For strings: offsets pointer
        gamax::size_type lhs_idx;
        gamax::size_type rhs_idx;
        
        template<typename T>
        __device__ __forceinline__
        int operator()() const
        {
            if constexpr (cuda::std::is_same_v<T, gamax::string_type_tag>) {
                // String comparison using ARROW format
                return compare_strings(
                    static_cast<const char*>(data),
                    static_cast<const gamax::size_type*>(meta),
                    lhs_idx, rhs_idx);
            } else {
                // Numeric comparison
                const T* typed_data = static_cast<const T*>(data);
                T lhs = typed_data[lhs_idx];
                T rhs = typed_data[rhs_idx];
                if (lhs < rhs) return -1;
                if (lhs > rhs) return 1;
                return 0;
            }
        }
    };
    
    // Multi-key row comparator using type_dispatcher_all
    struct MultiKeyRowComparator {
        KeyColumnInfo keys[MAX_SORT_KEYS];
        int num_keys;
        
        __host__ __device__
        MultiKeyRowComparator(const KeyColumnInfo* key_infos, int n_keys)
            : num_keys(n_keys)
        {
            for (int i = 0; i < n_keys && i < MAX_SORT_KEYS; ++i) {
                keys[i] = key_infos[i];
            }
        }
        
        __device__
        bool operator()(gamax::size_type lhs_idx, gamax::size_type rhs_idx) const
        {
            for (int k = 0; k < num_keys; ++k) {
                CompareElementsFunctor cmp_func{
                    keys[k].data, 
                    keys[k].meta, 
                    lhs_idx, 
                    rhs_idx
                };
                
                int cmp = gamax::type_dispatcher_all(keys[k].type, cmp_func);
                
                if (cmp != 0) {
                    // Apply sort order: ascending means lhs < rhs returns true
                    // descending means lhs > rhs returns true (flip the comparison)
                    return keys[k].is_ascending ? (cmp < 0) : (cmp > 0);
                }
            }
            
            // All keys equal - stable sort: preserve original order
            return lhs_idx < rhs_idx;
        }
    };

    // =========================================================================
    // Multi-key sorting implementation
    // =========================================================================
    std::unique_ptr<gamax::table>
    sort_multi_key_table(
        gamax::table_view const& table,
        std::vector<int> const& sort_keys,
        std::vector<gamax::sort_order> const& sort_orders,
        cudaStream_t stream)
    {
        const auto num_cols = table.num_columns();
        const auto num_rows = table.num_rows();
        const auto num_keys = static_cast<int>(sort_keys.size());
        
        if (num_rows == 0) {
            return make_empty_table_like(table, stream);
        }
        
        if (num_keys > MAX_SORT_KEYS) {
            throw std::invalid_argument("Too many sort keys (max " + std::to_string(MAX_SORT_KEYS) + ")");
        }
        
        // Build key column info on host
        std::vector<KeyColumnInfo> host_key_infos(num_keys);
        
        // We need to extract string column data/meta pointers
        // For string columns in a table_view, we need to copy the helper buffer to get the pointers
        std::vector<void*> string_ptrs_storage(num_keys * 2);
        
        for (int k = 0; k < num_keys; ++k) {
            auto col_view = table.get_column(sort_keys[k]);
            host_key_infos[k].type = col_view.type().id();
            host_key_infos[k].is_ascending = (sort_orders[k] == gamax::sort_order::ASCENDING);
            
            if (col_view.type().id() == gamax::type_id::STRING) {
                // For string columns, extract data and meta pointers from helper buffer
                void* ptrs[2];
                cudaMemcpyAsync(ptrs, col_view.data<void>(), 2 * sizeof(void*), 
                               cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                host_key_infos[k].data = ptrs[0];
                host_key_infos[k].meta = ptrs[1];
            } else {
                host_key_infos[k].data = col_view.data<void>();
                host_key_infos[k].meta = nullptr;
            }
        }
        
        // Create comparator
        MultiKeyRowComparator comparator(host_key_infos.data(), num_keys);
        
        // Initialize indices (0 to N-1)
        auto indices = gamax::make_sequence<gamax::size_type>(num_rows, 0, 1, stream);
        
        // Sort indices using multi-key comparator
        CALL_CUB_FUNC(
            cub::DeviceMergeSort::SortKeys,
            indices->template data<gamax::size_type>(),
            num_rows,
            comparator,
            stream);
        
        cudaStreamSynchronize(stream);
        
        // Now we have sorted indices. Apply them to all columns.
        // Create inverse permutation for efficient numeric column sorting
        auto indices_B = gamax::make_sequence<gamax::size_type>(num_rows, 0, 1, stream);
        auto [sorted_indices, inverse_perm] = sort_column_pair(
            indices->view(), indices_B->view(), 0, -1,
            gamax::sort_order::ASCENDING, nullptr, 0, stream);
        
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
            auto [_, sorted_col] = sort_column_pair(
                inverse_perm->view(),
                table.get_column(col_idx),
                0, -1, gamax::sort_order::ASCENDING, nullptr, 0, stream);
            sorted_columns[col_idx] = std::move(sorted_col);
        }
        
        // For string columns: gather using original sorted indices
        for (auto col_idx : string_col_indices) {
            auto col_view = table.get_column(col_idx);
            gamax::string_column_view str_view(col_view, gamax::ARROW_STRING, stream);
            sorted_columns[col_idx] = gamax::gather_strings(str_view, indices->view(), stream);
        }
        
        return gamax::make_table(std::move(sorted_columns));
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