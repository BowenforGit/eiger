// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "expr/expr_fast.hpp"

namespace gamax::select {
    struct select_options {
        bool fuse_operations = true;
    };

    std::unique_ptr<gamax::table> select(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream,
        select_options options = {});
    
    std::pair<std::unique_ptr<gamax::table>, gamax::column_ptr> select_and_return_bitmask(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream,
        select_options options = {});
    
    gamax::column_ptr select_to_bitmask(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream,
        select_options options = {});

    std::pair<std::unique_ptr<gamax::table>, gamax::column_ptr> select_fuse(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream);
    
    std::pair<std::unique_ptr<gamax::table>, gamax::column_ptr> select_nofuse(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream);

    // Forward declaration — full definition in predicate.hpp
    class Predicate;

    // `options` controls the fuse/nofuse algorithm of EXPR leaves
    // (numeric subtrees) inside the Predicate. STRING leaves carry
    // their own per-leaf string_op_options and aren't affected.
    gamax::column_ptr select_to_bitmask(
        gamax::table_view table,
        Predicate& predicate,
        cudaStream_t stream,
        select_options options = {});

} // namespace gamax::select
