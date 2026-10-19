// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "select.hpp"

#include <memory>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "expr/expr_fast.hpp"
#include "utils/transform.hpp"
#include "utils/utils.cuh"

namespace gamax::select {
std::unique_ptr<gamax::table> select(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream,
        select_options options)
{
    auto [result_table, _] = select_and_return_bitmask(table, expr, stream, options);
    return std::move(result_table);
}

std::pair<std::unique_ptr<gamax::table>, gamax::column_ptr> select_and_return_bitmask(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream,
        select_options options)
{
    if (options.fuse_operations) {
        return select_fuse(table, expr, stream);
    } else {
        return select_nofuse(table, expr, stream);
    }
}

gamax::column_ptr select_to_bitmask(
    gamax::table_view table,
    gamax::expression::ExprTree& expr,
    cudaStream_t stream,
    select_options options)
{
    if (options.fuse_operations) {
        return gamax::expression::evaluate_expr(table, expr, stream);;
    } else {
        return gamax::expression::evaluate_expr_nofuse(table, expr, stream);
    }
}

std::pair<std::unique_ptr<gamax::table>, gamax::column_ptr> select_fuse(
    gamax::table_view table,
    gamax::expression::ExprTree& expr,
    cudaStream_t stream) 
{
    auto mask = gamax::expression::evaluate_expr(table, expr, stream);
    return {gamax::util::device::compact(table, mask->view(), stream), std::move(mask)};
}

std::pair<std::unique_ptr<gamax::table>, gamax::column_ptr> select_nofuse(
    gamax::table_view table,
    gamax::expression::ExprTree& expr,
    cudaStream_t stream) 
{
    auto mask = gamax::expression::evaluate_expr_nofuse(table, expr, stream);
    return {gamax::util::device::compact(table, mask->view(), stream), std::move(mask)};
}
} // namespace gamax::select
