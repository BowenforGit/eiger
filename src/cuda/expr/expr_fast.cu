// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <memory>
#include <vector>
#include <variant>
#include <utility>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "memory/device_buffer.hpp"
#include "expr/expr_fast.hpp"
#include "expr/expr_impl.cuh"
#include "memory/scalar.hpp"
#include "utils/utils.cuh"

namespace gamax::expression {
namespace {
    using expr_program_ptr_t = decltype(std::declval<ExprTree>().make_program(cudaStreamDefault));
}

namespace detail::common {
struct expr_intermediate_storage_regs {
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

struct expr_intermediate_storage_shared {
    gamax::scalar::scalar_data_type* data;
    gamax::type_id* types;
    int stride;   // max_regs per thread

    __device__ __forceinline__ gamax::scalar::scalar_data_type& data_ref(int idx) {
        return data[stride * static_cast<int>(threadIdx.x) + idx];
    }
    __device__ __forceinline__ gamax::type_id& type_ref(int idx) {
        return types[stride * static_cast<int>(threadIdx.x) + idx];
    }

    __device__ __forceinline__ void store(int idx, gamax::scalar const& s) {
        data_ref(idx) = s._data;
        type_ref(idx) = s._type;
    }

    __device__ __forceinline__ gamax::scalar load(int idx) {
        return gamax::scalar(type_ref(idx), data_ref(idx));
    }
};

template <typename Storage>
__device__ void evaluate_one(
    gamax::experimental::table_device_view table,
    Expr& e,
    gamax::size_type i,
    Storage& s)
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
        case ExprId::Sqrt: {
            auto op1 = s.load(e.operand_regs[0]);
            s.store(
                e.return_reg,
                gamax::type_dispatcher(
                    op1.type(),
                    detail::sqrt_helper{},
                    op1));
            return;
        }
        case ExprId::Log: {
            auto op1 = s.load(e.operand_regs[0]);
            s.store(
                e.return_reg,
                gamax::type_dispatcher(
                    op1.type(),
                    detail::log_helper{},
                    op1));
            return;
        }
        case ExprId::Exp: {
            auto op1 = s.load(e.operand_regs[0]);
            s.store(
                e.return_reg,
                gamax::type_dispatcher(
                    op1.type(),
                    detail::exp_helper{},
                    op1));
            return;
        }
        case ExprId::YearOf: {
            auto op1 = s.load(e.operand_regs[0]);
            s.store(
                e.return_reg,
                gamax::type_dispatcher(
                    op1.type(),
                    detail::year_of_helper{},
                    op1));
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

template <typename Storage>
__device__ gamax::scalar evaluate(
    gamax::experimental::table_device_view table,
    ExprProgram           ep,
    gamax::size_type      i,
    Storage& storage)
{
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
        expr_intermediate_storage_regs storage;
        gamax::scalar res = evaluate(table, ep, idx, storage);
        output[idx] = res.get<T>();
    }
}

template<typename T>
__global__ void evaluate_expr_kernel_shared(
    gamax::experimental::table_device_view table,
    ExprProgram ep,
    gamax::size_type nrows,
    T* output,
    int max_regs)
{
    extern __shared__ unsigned char smem[];
    auto* data = reinterpret_cast<gamax::scalar::scalar_data_type*>(smem);
    auto* types = reinterpret_cast<gamax::type_id*>(data + max_regs * blockDim.x);
    expr_intermediate_storage_shared storage{data, types, max_regs};

    for(std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < nrows; 
        idx += blockDim.x * gridDim.x) 
    {
        gamax::scalar res = evaluate(table, ep, idx, storage);
        output[idx] = res.get<T>();
    }
}

struct project_col_impl_reg {
    template<typename T>
    std::unique_ptr<gamax::column> operator()(
        gamax::table_view& table,
        expr_program_ptr_t expr_program_ptr,
        cudaStream_t stream)
    {
        gamax::size_type nrows = table.num_rows();
        auto col = gamax::make_column<T>(nrows, stream);

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

struct project_col_impl_shared {
    template<typename T>
    std::unique_ptr<gamax::column> operator()(
        gamax::table_view& table,
        expr_program_ptr_t expr_program_ptr,
        int max_regs,
        cudaStream_t stream)
    {
        gamax::size_type nrows = table.num_rows();
        auto col = gamax::make_column<T>(nrows, stream);

        auto device_table_ptr = table.device_view(stream);

        auto expr_program = *expr_program_ptr;
        auto device_table = *device_table_ptr;

        const int block = 128;
        std::size_t per_thread = static_cast<std::size_t>(max_regs) * (sizeof(gamax::scalar::scalar_data_type) + sizeof(gamax::type_id));
        std::size_t shmem_bytes = per_thread * block;

#ifdef __NVCC__
        if (shmem_bytes > 48 * 1024) {
            CHECK_CUDA_ERROR(cudaFuncSetAttribute(
                evaluate_expr_kernel_shared<T>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem_bytes)));
        }
#endif

        evaluate_expr_kernel_shared<<<num_tb(nrows, block), block, shmem_bytes, stream>>>(
            device_table,
            expr_program,
            nrows,
            col->template data<T>(),
            max_regs);
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
struct expr_intermediate_storage_shared {
    T* data;
    int stride;

    __device__ __forceinline__ T& ref(int idx) {
        return data[stride * static_cast<int>(threadIdx.x) + idx];
    }

    __device__ __forceinline__ void store(int idx, T const& v) { ref(idx) = v; }
    __device__ __forceinline__ T load(int idx) { return ref(idx); }
};

template<typename T, typename Storage>
__device__ void evaluate_one(
    gamax::experimental::table_device_view table,
    Expr& e,
    gamax::size_type i,
    Storage& s)
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
        case ExprId::Sqrt: {
            auto op1 = s.load(e.operand_regs[0]);
            if constexpr (std::is_same<T, float>::value) {
                s.store(e.return_reg, sqrtf(op1));
            } else if constexpr (std::is_same<T, double>::value) {
                s.store(e.return_reg, sqrt(op1));
            } else {
                s.store(e.return_reg, static_cast<T>(sqrt(static_cast<double>(op1))));
            }
            return;
        }
        case ExprId::Log: {
            auto op1 = s.load(e.operand_regs[0]);
            if constexpr (std::is_same<T, float>::value) {
                s.store(e.return_reg, logf(op1));
            } else if constexpr (std::is_same<T, double>::value) {
                s.store(e.return_reg, log(op1));
            } else {
                s.store(e.return_reg, static_cast<T>(log(static_cast<double>(op1))));
            }
            return;
        }
        case ExprId::Exp: {
            auto op1 = s.load(e.operand_regs[0]);
            if constexpr (std::is_same<T, float>::value) {
                s.store(e.return_reg, expf(op1));
            } else if constexpr (std::is_same<T, double>::value) {
                s.store(e.return_reg, exp(op1));
            } else {
                s.store(e.return_reg, static_cast<T>(exp(static_cast<double>(op1))));
            }
            return;
        }
        case ExprId::YearOf: {
            if constexpr (!std::is_same<T, int32_t>::value) {
                GAMAX_UNREACHABLE("year_of only supports int32 (YYYYMMDD) inputs.");
            } else {
                auto op1 = s.load(e.operand_regs[0]);
                s.store(e.return_reg, op1 / 10000);
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

template<typename T, typename Storage>
__device__ T evaluate(
    gamax::experimental::table_device_view table,
    ExprProgram           ep,
    gamax::size_type      i,
    Storage& storage)
{
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
        expr_intermediate_storage<T> storage;
        auto res = evaluate<T>(table, ep, idx, storage);
        output[idx] = static_cast<RetType>(res);
    }
}

template<typename T, typename RetType>
__global__ void evaluate_expr_kernel_shared(
    gamax::experimental::table_device_view table,
    ExprProgram ep,
    gamax::size_type nrows,
    RetType* output,
    int max_regs)
{
    extern __shared__ unsigned char smem[];
    auto* data = reinterpret_cast<T*>(smem);
    expr_intermediate_storage_shared<T> storage{data, max_regs};

    for(std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < nrows; 
        idx += blockDim.x * gridDim.x) 
    {
        auto res = evaluate<T>(table, ep, idx, storage);
        output[idx] = static_cast<RetType>(res);
    }
}

struct project_col_impl_reg {
    template<typename T, typename RetType>
    std::unique_ptr<gamax::column> operator()(
        gamax::table_view& table,
        expr_program_ptr_t expr_program_ptr,
        cudaStream_t stream)
    {
        gamax::size_type nrows = table.num_rows();
        auto col = gamax::make_column<RetType>(nrows, stream);

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

struct project_col_impl_shared {
    template<typename T, typename RetType>
    std::unique_ptr<gamax::column> operator()(
        gamax::table_view& table,
        expr_program_ptr_t expr_program_ptr,
        int max_regs,
        cudaStream_t stream)
    {
        gamax::size_type nrows = table.num_rows();
        auto col = gamax::make_column<RetType>(nrows, stream);

        auto device_table_ptr = table.device_view(stream);

        auto expr_program = *expr_program_ptr;
        auto device_table = *device_table_ptr;

        const int block = 128;
        std::size_t per_thread = static_cast<std::size_t>(max_regs) * sizeof(T);
        std::size_t shmem_bytes = per_thread * block;

#ifdef __NVCC__
        if (shmem_bytes > 48 * 1024) {
            CHECK_CUDA_ERROR(cudaFuncSetAttribute(
                evaluate_expr_kernel_shared<T, RetType>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem_bytes)));
        }
#endif

        evaluate_expr_kernel_shared<T, RetType><<<num_tb(nrows, block), block, shmem_bytes, stream>>>(
            device_table, 
            expr_program, 
            nrows,
            col->template data<RetType>(),
            max_regs);
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
    auto expr_program_ptr = et.make_program(stream);

    int max_regs = et.max_regs();
    bool use_shared = max_regs > MAX_EXPR_REGISTERS;

    auto common_type = et.get_common_type();
    if (common_type != gamax::type_id::EMPTY) {
        if (use_shared) {
            return gamax::double_type_dispatcher(
                gamax::data_type(common_type),
                et.type(),
                detail::fast_path::project_col_impl_shared{},
                table,
                std::move(expr_program_ptr),
                max_regs,
                stream);
        } else {
            return gamax::double_type_dispatcher(
                gamax::data_type(common_type),
                et.type(),
                detail::fast_path::project_col_impl_reg{},
                table,
                std::move(expr_program_ptr),
                stream);
        }
    } else {
        if (use_shared) {
            return gamax::type_dispatcher(
                et.type(),
                detail::common::project_col_impl_shared{},
                table,
                std::move(expr_program_ptr),
                max_regs,
                stream);
        } else {
            return gamax::type_dispatcher(
                et.type(),
                detail::common::project_col_impl_reg{},
                table,
                std::move(expr_program_ptr),
                stream);
        }
    }
}
} // namespace gamax
