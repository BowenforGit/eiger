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
#include "expr/expr_fast.cuh"
#include "expr/expr_impl.cuh"
#include "memory/scalar.hpp"
#include "utils/utils.cuh"

namespace gamax::expression {
namespace {
    gamax::type_id is_fast_path_compatible(ExprTree const& et) {
        // TODO: implement checks for fast path compatibility
        return gamax::type_id::EMPTY;
    }
} // namespace anonymous

namespace detail::common {
struct expr_intermediate_storage {
    gamax::scalar::scalar_data_type d0, d1, d2;
    gamax::type_id t0, t1, t2;

    __device__ __forceinline__ void store(int idx, gamax::scalar const& s) {
        switch(idx) {
            case 0: d0 = s._data; t0 = s._type; break;
            case 1: d1 = s._data; t1 = s._type; break;
            case 2: d2 = s._data; t2 = s._type; break;
            default: GAMAX_UNREACHABLE();
        }
    }

    __device__ __forceinline__ gamax::scalar load(int idx) {
        switch(idx) {
            case 0: return gamax::scalar(t0, d0);
            case 1: return gamax::scalar(t1, d1);
            case 2: return gamax::scalar(t2, d2);
            default: GAMAX_UNREACHABLE();
        }
    }
};

__device__ void evaluate_one(
    gamax::experimental::table_device_view table,
    Expr& e,
    gamax::size_type i,
    expr_intermediate_storage& s)
{
    switch(e.id) {
        case ExprId::Constant: {
            return s.store(e.return_reg, e.data.c.value);
        }
        case ExprId::Column: {
            auto col = table[e.data.col.cid];
            auto value = gamax::type_dispatcher(
                col.type(),
                detail::column_access_helper{},
                col,
                i);
            s.store(e.return_reg, value);
            return;
        }
        case ExprId::Negate: {
            auto op1 = s.load(e.operand_regs[0]);
            s.store(
                e.return_reg,
                gamax::type_dispatcher(
                    op1.type(),
                    detail::negate_helper{},
                    op1));
            return;
        }
        case ExprId::LeftShift: {
            auto op1 = s.load(e.operand_regs[0]);
            s.store(
                e.return_reg,
                gamax::type_dispatcher(
                    op1.type(),
                    detail::left_shift_helper{},
                    op1,
                    e.data.lshift.bits));
            return;
        }
        case ExprId::RightShift: {
            auto op1 = s.load(e.operand_regs[0]);
            s.store(
                e.return_reg,
                gamax::type_dispatcher(
                    op1.type(),
                    detail::right_shift_helper{},
                    op1,
                    e.data.rshift.bits));
            return;
        }
        case ExprId::Add: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::add_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::Substract: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::substract_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::Multiply: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::multiply_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::Divide: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::divide_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::GreaterThan: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::gt_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::LessThan: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::lt_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::GreaterEqual: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::ge_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::LessEqual: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::le_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::Equal: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::eq_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::And: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::and_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::Or: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op1.type(),
                    op2.type(),
                    detail::or_helper{},
                    op1,
                    op2));
            return;
        }
        case ExprId::Select: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            auto op3 = s.load(e.operand_regs[2]);
            s.store(
                e.return_reg,
                gamax::double_type_dispatcher(
                    op2.type(),
                    op3.type(),
                    detail::select_helper{},
                    op1,
                    op2,
                    op3));
            return;
        }
        default:
            GAMAX_UNREACHABLE();
    }
}

__device__ gamax::scalar evaluate(
    gamax::experimental::table_device_view table,
    ExprProgram           ep,
    gamax::size_type      i)
{
    expr_intermediate_storage storage;
    
    // Execute each expression in the program sequentially
    for (gamax::size_type j = 0; j < ep.n_exprs; ++j) {
        evaluate_one(table, ep.exprs[j], i, storage);
    }
    
    // The final result is in the return register of the last expression
    return storage.load(ep.exprs[ep.n_exprs - 1].return_reg);
}

template<typename T>
__global__ void evaluate_expr_kernel(
    gamax::experimental::table_device_view table,
    ExprProgram ep,
    gamax::size_type nrows,
    T* output)
{
    for(std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < nrows; 
        idx += blockDim.x * gridDim.x) 
    {
        gamax::scalar res = evaluate(table, ep, idx);
        output[idx] = res.get<T>();
    }
}

struct project_col_impl {
    template<typename T>
    std::unique_ptr<gamax::column> operator()(
        gamax::table_view& table,
        ExprTree& et,
        cudaStream_t stream)
    {
        gamax::size_type nrows = table.num_rows();
        auto col = gamax::make_column<T>(nrows);

        auto expr_program_ptr = et.make_program(stream);
        auto device_table_ptr = table.device_view(stream);

        auto expr_program = *expr_program_ptr;
        auto device_table = *device_table_ptr;
        
        evaluate_expr_kernel<<<num_tb(nrows, 128), 128, 0, stream>>>(
            device_table, 
            expr_program, 
            nrows,
            col->template data<T>());
        cudaStreamSynchronize(stream);
        
        return col;
    }
};
}
namespace detail::fast_path {

template<typename T>
struct expr_intermediate_storage {
    T d0, d1, d2;

    __device__ __forceinline__ void store(int idx, T const& s) {
        switch(idx) {
            case 0: d0 = s; break;
            case 1: d1 = s; break;
            case 2: d2 = s; break;
            // case 3: d3 = s._data; t3 = s._type; break;
            default: GAMAX_UNREACHABLE();
        }
    }

    __device__ __forceinline__ T load(int idx) {
        switch(idx) {
            case 0: return d0;
            case 1: return d1;
            case 2: return d2;
            // case 3: return gamax::scalar(t3, d3);
            default: GAMAX_UNREACHABLE();
        }
    }
};

template<typename T>
__device__ void evaluate_one(
    gamax::experimental::table_device_view table,
    Expr& e,
    gamax::size_type i,
    expr_intermediate_storage<T>& s)
{
    switch(e.id) {
        case ExprId::Constant: {
            return s.store(e.return_reg, e.data.c.value.get<T>());
        }
        case ExprId::Column: {
            auto col = table[e.data.col.cid];
            s.store(e.return_reg, col.data<T>()[i]);
            return;
        }
        case ExprId::Negate: {
            auto op1 = s.load(e.operand_regs[0]);
            s.store(e.return_reg, -op1);
            return;
        }
        case ExprId::LeftShift: {
            if constexpr (!std::is_integral<T>::value) {
                GAMAX_UNREACHABLE("Only integral types can be left-shifted.");
            } else {
                auto op1 = s.load(e.operand_regs[0]);
                s.store(
                    e.return_reg,
                    op1 << e.data.lshift.bits);
            }
            return;
        }
        case ExprId::RightShift: {
            if constexpr (!std::is_integral<T>::value) {
                GAMAX_UNREACHABLE("Only integral types can be right-shifted.");
            } else {
                auto op1 = s.load(e.operand_regs[0]);
                s.store(
                    e.return_reg,
                    op1 >> e.data.rshift.bits);
            }
            return;
        }
        case ExprId::Add: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 + op2);
            return;
        }
        case ExprId::Substract: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 - op2);
            return;
        }
        case ExprId::Multiply: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 * op2);
            return;
        }
        case ExprId::Divide: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 / op2);
            return;
        }
        case ExprId::GreaterThan: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 > op2);
            return;
        }
        case ExprId::LessThan: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 < op2);
            return;
        }
        case ExprId::GreaterEqual: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 >= op2);
            return;
        }
        case ExprId::LessEqual: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 <= op2);
            return;
        }
        case ExprId::Equal: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 == op2);
            return;
        }
        case ExprId::And: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 && op2);
            return;
        }
        case ExprId::Or: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            s.store(
                e.return_reg,
                op1 || op2);
            return;
        }
        case ExprId::Select: {
            auto op1 = s.load(e.operand_regs[0]);
            auto op2 = s.load(e.operand_regs[1]);
            auto op3 = s.load(e.operand_regs[2]);
            s.store(
                e.return_reg,
                op1 ? op2 : op3);
            return;
        }
        default:
            GAMAX_UNREACHABLE();
    }
}

template<typename T>
__device__ T evaluate(
    gamax::experimental::table_device_view table,
    ExprProgram           ep,
    gamax::size_type      i)
{
    expr_intermediate_storage<T> storage;
    
    // Execute each expression in the program sequentially
    for (gamax::size_type j = 0; j < ep.n_exprs; ++j) {
        evaluate_one<T>(table, ep.exprs[j], i, storage);
    }
    
    // The final result is in the return register of the last expression
    return storage.load(ep.exprs[ep.n_exprs - 1].return_reg);
}

template<typename T, typename RetType>
__global__ void evaluate_expr_kernel(
    gamax::experimental::table_device_view table,
    ExprProgram ep,
    gamax::size_type nrows,
    RetType* output)
{
    for(std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < nrows; 
        idx += blockDim.x * gridDim.x) 
    {
        auto res = evaluate<T>(table, ep, idx);
        output[idx] = static_cast<RetType>(res);
    }
}

struct project_col_impl {
    template<typename T, typename RetType>
    std::unique_ptr<gamax::column> operator()(
        gamax::table_view& table,
        ExprTree& et,
        cudaStream_t stream)
    {
        gamax::size_type nrows = table.num_rows();
        auto col = gamax::make_column<RetType>(nrows);

        auto expr_program_ptr = et.make_program(stream);
        auto device_table_ptr = table.device_view(stream);

        auto expr_program = *expr_program_ptr;
        auto device_table = *device_table_ptr;
        
        evaluate_expr_kernel<T, RetType><<<num_tb(nrows, 128), 128, 0, stream>>>(
            device_table, 
            expr_program, 
            nrows,
            col->template data<RetType>());
        cudaStreamSynchronize(stream);
        
        return col;
    }
};
} // namespace detail::fast_path

std::unique_ptr<gamax::column> evaluate_expr(
    gamax::table_view table,
    ExprTree& et, 
    cudaStream_t stream) 
{   
    auto common_type = is_fast_path_compatible(et);
    if (common_type != gamax::type_id::EMPTY) {
        return gamax::double_type_dispatcher(
            gamax::data_type(common_type),
            et.type(),
            detail::fast_path::project_col_impl{},
            table,
            et,
            stream);
    } else {
        return gamax::type_dispatcher(
            et.type(),
            detail::common::project_col_impl{},
            table,
            et,
            stream);
    }
}
} // namespace gamax