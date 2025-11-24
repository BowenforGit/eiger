#pragma once

#include <memory>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "expr/expr_fast.cuh"

namespace gamax::select {
    std::unique_ptr<gamax::table> select(
        gamax::table_view table,
        gamax::expression::ExprTree& expr,
        cudaStream_t stream = cudaStreamDefault);
} // namespace gamax::select