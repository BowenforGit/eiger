// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

#include "types/data_type.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "utils/transform.hpp"

namespace gamax::sort {
namespace detail {

// =========================================================================
// Helper: Create empty table with same schema
// =========================================================================
inline std::unique_ptr<gamax::table>
make_empty_table_like(gamax::table_view const& table, cudaStream_t stream)
{
    std::vector<std::unique_ptr<gamax::column>> empty_columns;
    const auto num_cols = table.num_columns();
    for (gamax::size_type i = 0; i < num_cols; ++i) {
        if (table.get_column(i).type().id() == gamax::type_id::STRING) {
            empty_columns.push_back(std::make_unique<gamax::string_column>(
                gamax::device_buffer{0, stream}, gamax::device_buffer{0, stream},
                0, gamax::ARROW_STRING));
        } else {
            empty_columns.push_back(gamax::make_column(table.get_column(i).type(), 0, stream));
        }
    }
    return gamax::make_table(std::move(empty_columns));
}

// =========================================================================
// Helper: Apply sorted indices to reorder all columns using gather
// =========================================================================
inline void gather_columns_by_indices(
    gamax::table_view const& table,
    gamax::column_view const& sorted_indices,
    std::vector<std::unique_ptr<gamax::column>>& sorted_columns,
    cudaStream_t stream)
{
    const auto num_cols = table.num_columns();
    
    for (gamax::size_type i = 0; i < num_cols; ++i) {
        auto col = table.get_column(i);
        
        if (col.type().id() == gamax::type_id::STRING) {
            gamax::string_column_view str_view(col, gamax::ARROW_STRING);
            sorted_columns[i] = gamax::gather_strings(str_view, sorted_indices, stream);
        } else {
            sorted_columns[i] = gamax::util::device::gather(sorted_indices, col, stream);
        }
    }
}

// =========================================================================
// Helper: Create sorted table from indices
// =========================================================================
inline std::unique_ptr<gamax::table>
create_sorted_table_from_indices(
    gamax::table_view const& table,
    gamax::column_view const& sorted_indices,
    cudaStream_t stream)
{
    const auto num_cols = table.num_columns();
    std::vector<std::unique_ptr<gamax::column>> sorted_columns(num_cols);
    gather_columns_by_indices(table, sorted_indices, sorted_columns, stream);
    return gamax::make_table(std::move(sorted_columns));
}

} // namespace detail
} // namespace gamax::sort
