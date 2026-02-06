#include "operators/project/project.hpp"

#include <memory>
#include <vector>
#include <unordered_set>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "expr/expr_fast.hpp"
#include "utils/utils.cuh"
#include "utils/copying.hpp"

namespace gamax::project {
    std::unique_ptr<gamax::table> project(
        gamax::table_view table,
        std::vector<gamax::expression::ExprTree>& expr,
        project_options options,
        cudaStream_t stream)
    {
        if (options.fuse_operations) {
            return project_fuse(table, expr, stream);
        } else {
            return project_nofuse(table,  expr, stream);
        }
    }

    std::unique_ptr<gamax::table> project_fuse(
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

    std::unique_ptr<gamax::table> project(
        std::unique_ptr<gamax::table> input_table,
        std::vector<gamax::expression::ExprTree>& expr,
        project_options options,
        cudaStream_t stream)
    {
        // Create a view before we start moving columns
        // This view is used for evaluating complex expressions
        gamax::table_view table_view = input_table->view();

        // Track which columns have been moved (for handling duplicate references)
        std::unordered_set<gamax::size_type> moved_columns;

        std::vector<column_ptr> projected_columns;
        projected_columns.reserve(expr.size());

        for (auto& e : expr) {
            if (e.is_column_ref()) {
                gamax::size_type col_idx = e.get_column_index();
                
                if (moved_columns.find(col_idx) == moved_columns.end()) {
                    // First reference to this column: move it (zero-copy)
                    auto col = input_table->release_column(col_idx);
                    projected_columns.push_back(std::move(col));
                    moved_columns.insert(col_idx);
                } else {
                    // Column already moved: copy via device-to-device memcpy
                    auto col = gamax::copy_column(table_view[col_idx], stream);
                    projected_columns.push_back(std::move(col));
                }
            } else {
                // Complex expression: evaluate using existing functions
                if (options.fuse_operations) {
                    auto col = gamax::expression::evaluate_expr(table_view, e, stream);
                    projected_columns.push_back(std::move(col));
                } else {
                    auto col = gamax::expression::evaluate_expr_nofuse(table_view, e, stream);
                    projected_columns.push_back(std::move(col));
                }
            }
        }

        return gamax::make_table(std::move(projected_columns));
    }
} // namespace gamax::project