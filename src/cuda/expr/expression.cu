#include <memory>
#include <vector>
#include <variant>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "memory/device_buffer.hpp"
#include "expr/expression.cuh"
#include "expr/scalar.hpp"

namespace gamax::expression {
namespace {
__device__ gamax::scalar evaluate(
    ExprTreeDevice const& et, 
    gamax::size_type nid, 
    gamax::size_type i) 
{
    ExprNode const& e = et[nid];

    if(std::holds_alternative<ConstantExpr>(e.expr)) {
        return std::get<ConstantExpr>(e.expr).eval();
    }

    if(std::holds_alternative<ColumnExpr>(e.expr)) {
        return std::get<ColumnExpr>(e.expr).eval(i);
    }

    gamax::scalar c1_res = evaluate(et, e.children[0], i);
    if(std::holds_alternative<NegateExpr>(e.expr)) {
        return std::get<NegateExpr>(e.expr).eval(c1_res);
    } else if(std::holds_alternative<LeftShiftExpr>(e.expr)) {
        return std::get<LeftShiftExpr>(e.expr).eval(c1_res);
    } else {
        gamax::scalar c2_res = evaluate(et, e.children[1], i);
        if(std::holds_alternative<AddExpr>(e.expr)) {
            return std::get<AddExpr>(e.expr).eval(c1_res, c2_res);
        } else if(std::holds_alternative<SubstractExpr>(e.expr)) {
            return std::get<SubstractExpr>(e.expr).eval(c1_res, c2_res);
        } else if(std::holds_alternative<MultiplyExpr>(e.expr)) {
            return std::get<MultiplyExpr>(e.expr).eval(c1_res, c2_res);
        } else {
            GAMAX_UNREACHABLE();
        }
    }
}

struct project_col_impl {
    template<typename T>
    std::unique_ptr<gamax::column> operator()(
        ExprTree& et,
        cudaStream_t stream)
    {
        auto col = gamax::make_column<T>(et.nrows());

        auto device_et_ptr = et.to_device(stream);
        auto& device_et = *device_et_ptr;
        auto itr = thrust::make_counting_iterator<gamax::size_type>(0);
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            itr, itr+et.nrows(),
            col->template data<T>(),
            [device_et] __device__ (gamax::size_type idx) {
                return evaluate(device_et, device_et.root_id(), idx).get<T>();
            }
        );
        cudaStreamSynchronize(stream);
        
        return col;
    }
};
} // namespace

gamax::data_type get_expr_type(ExprType const& e) {
    return std::visit([](auto const& e) { return e.type(); }, e);
}

std::unique_ptr<gamax::column> evaluate_expr(
    ExprTree& et, 
    cudaStream_t stream) 
{   
    return gamax::type_dispatcher(get_expr_type(et.root().expr), project_col_impl{}, et, stream);
}
} // namespace gamax