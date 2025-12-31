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
        select_options options = {},
        cudaStream_t stream = cudaStreamDefault);

    std::unique_ptr<gamax::table> select_fuse(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream = cudaStreamDefault);
    
    std::unique_ptr<gamax::table> select_nofuse(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream = cudaStreamDefault);
} // namespace gamax::select