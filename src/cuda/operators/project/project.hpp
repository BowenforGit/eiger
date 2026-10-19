// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <vector>
#include <unordered_set>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "expr/expr_fast.hpp"

namespace gamax::project {
    struct project_options {
        bool fuse_operations = true;
    };

    std::unique_ptr<gamax::table> project(
        gamax::table_view table,
        std::vector<gamax::expression::ExprTree>& expr,
        cudaStream_t stream,
        project_options options = {});
    
    std::unique_ptr<gamax::table> project_fuse(
        gamax::table_view table,
        std::vector<gamax::expression::ExprTree>& expr,
        cudaStream_t stream);
    
    std::unique_ptr<gamax::table> project_nofuse(
        gamax::table_view table,
        std::vector<gamax::expression::ExprTree>& expr,
        cudaStream_t stream);
    
    /**
     * @brief Project with ownership transfer optimization.
     * 
     * This overload takes ownership of the input table, enabling zero-copy
     * column transfer when an expression is a simple column reference.
     * For complex expressions, falls back to evaluate_expr.
     * 
     * When the same column is referenced multiple times, the first reference
     * will be moved (zero-copy) and subsequent references will be copied.
     *
     * @param input_table The input table (ownership transferred)
     * @param expr Vector of expressions defining the projection
     * @param options Project options (fuse_operations flag)
     * @param stream CUDA stream for async execution
     * @return New table with projected columns
     */
    std::unique_ptr<gamax::table> project(
        std::unique_ptr<gamax::table> input_table,
        std::vector<gamax::expression::ExprTree>& expr,
        cudaStream_t stream,
        project_options options = {});
} // namespace gamax::project
