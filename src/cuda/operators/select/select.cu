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
    cudaStream_t stream) 
{
    auto mask = gamax::expression::evaluate_expr(table, expr, stream);
    return gamax::util::device::compact(table, mask->view(), stream);
}

std::unique_ptr<gamax::table> select_nofuse(
    gamax::table_view table,
    gamax::expression::ExprTree& expr,
    cudaStream_t stream) 
{
    auto mask = gamax::expression::evaluate_expr_nofuse(table, expr, stream);
    return gamax::util::device::compact(table, mask->view(), stream);
}
} // namespace gamax::select