#pragma once

#include <memory>
#include <vector>

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
        project_options options = {},
        cudaStream_t stream = cudaStreamDefault);
    
    std::unique_ptr<gamax::table> project_fuse(
        gamax::table_view table,
        std::vector<gamax::expression::ExprTree>& expr,
        cudaStream_t stream = cudaStreamDefault);
    
    std::unique_ptr<gamax::table> project_nofuse(
        gamax::table_view table,
        std::vector<gamax::expression::ExprTree>& expr,
        cudaStream_t stream = cudaStreamDefault);
} // namespace gamax::project