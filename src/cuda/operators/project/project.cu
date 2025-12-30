#include "operators/project/project.hpp"

#include <memory>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "expr/expr_fast.hpp"
#include "utils/utils.cuh"

namespace gamax::project {
    std::unique_ptr<gamax::table> project(
        gamax::table_view table,
        std::vector<gamax::expression::ExprTree>& expr,
        cudaStream_t stream)
    {
        std::vector<column_ptr> projected_columns;
        projected_columns.reserve(expr.size());
        for (auto& e : expr) {
            auto col = gamax::expression::evaluate_expr(table, e, stream);
            projected_columns.push_back(std::move(col));
        }

        return gamax::make_table(std::move(projected_columns));
    }
    
    std::unique_ptr<gamax::table> project_nofuse(
        gamax::table_view table,
        std::vector<gamax::expression::ExprTree>& expr,
        cudaStream_t stream)
    {
        std::vector<column_ptr> projected_columns;
        projected_columns.reserve(expr.size());
        for (auto& e : expr) {
            auto col = gamax::expression::evaluate_expr_nofuse(table, e, stream);
            projected_columns.push_back(std::move(col));
        }

        return gamax::make_table(std::move(projected_columns));
    }
} // namespace gamax::project