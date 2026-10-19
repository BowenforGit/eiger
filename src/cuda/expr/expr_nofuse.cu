// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <memory>
#include <vector>
#include <variant>
#include <unordered_map>
#include <type_traits>
#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "memory/device_buffer.hpp"
#include "expr/expr_fast.hpp"
#include "expr/expr_impl.cuh"
#include "memory/scalar.hpp"
#include "utils/copying.hpp"
#include "utils/utils.cuh"
#include "utils/green_safe_elementwise.cuh"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "types/vec_type.hpp"

namespace gamax::expression {
namespace {

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

struct intermediate_storage {
    using operand_type = std::variant<
        gamax::column_ptr, // intermediate results stored as column pointers
        gamax::column_view, // columns from the input table
        gamax::scalar // scalar constants
    >; 
    
    operand_type load(int key) {
        return std::move(_storage.at(key));
    }
    
    void store(int key, operand_type value) {
        _storage[key] = std::move(value);
    }
    
private:
    std::unordered_map<int, operand_type> _storage;
};

template <typename T>
struct column_view_itr {
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::random_access_iterator_tag;

    column_view_itr(column_view& col) : _data(col.data()) {}

    __host__ __device__ auto operator[](size_type idx) const {
        return static_cast<const T*>(_data)[idx];
    }
    
    // Vectorized load interface
    template<int VT>
    __host__ __device__ auto load_vec(size_type idx) const {
        using VecType = typename vec_type<T, VT>::type;
        const T* ptr = static_cast<const T*>(_data);
        return reinterpret_cast<const VecType*>(ptr)[idx / VT];
    }

private:
    // column_view _col;
    const void* _data;
};

// Custom constant iterator with vectorized load support
template <typename T>
struct constant_itr {
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;
    using iterator_category = std::random_access_iterator_tag;

    constant_itr(T value) : _value(value) {}

    __host__ __device__ T operator[](size_type idx) const {
        return _value;
    }
    
    // Vectorized load interface - constructs a vector with all components set to the constant
    template<int VT>
    __host__ __device__ auto load_vec(size_type idx) const {
        using VecType = typename vec_type<T, VT>::type;
        VecType vec;
        if constexpr (VT >= 1) vec.x = _value;
        if constexpr (VT >= 2) vec.y = _value;
        if constexpr (VT >= 3) vec.z = _value;
        if constexpr (VT >= 4) vec.w = _value;
        return vec;
    }

private:
    T _value;
};

template<typename T>
column_view_itr<T> make_iterator(column_view col) {
    return column_view_itr<T>(col);
}

template<typename T>
constant_itr<T> make_iterator(gamax::scalar s) {
    return constant_itr<T>(s.get<T>());
}

template <typename InputItr, typename Op>
__global__ void kernel_unary_op(
    InputItr input,
    mutable_column_view output,
    Op op)
{
    using T1 = typename std::iterator_traits<InputItr>::value_type;
    using ResType = std::remove_cv_t<std::remove_reference_t<decltype(op(cuda::std::declval<T1>()))>>;
    
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type stride = blockDim.x * gridDim.x;
    const size_type nrows = output.size();

    #pragma unroll
    for (size_type i = idx; i < nrows; i += stride) {
        output.data<ResType>()[i] = op(input[i]);
    }
}

// Unified vectorized kernel using iterator load_vec interface
template <typename InputItr, typename Op, int VT = 4>
__global__ void kernel_unary_op_vec(
    InputItr input,
    mutable_column_view output,
    Op op)
{
    using T1 = typename std::iterator_traits<InputItr>::value_type;
    using ResType = std::remove_cv_t<std::remove_reference_t<decltype(op(cuda::std::declval<T1>()))>>;
    
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type stride = blockDim.x * gridDim.x;
    const size_type nrows = output.size();
    const size_type vec_nrows = (nrows / VT) * VT;  // Number of elements that fit in complete vectors

    using VecTypeIn = typename vec_type<T1, VT>::type;
    using VecTypeOut = typename vec_type<ResType, VT>::type;
    ResType* output_ptr = output.data<ResType>();
    
    // Vectorized processing - each thread processes VT elements at a time
    #pragma unroll
    for (size_type i = idx * VT; i < vec_nrows; i += stride * VT) {
        // Vectorized load via iterator interface
        VecTypeIn vec_in = input.template load_vec<VT>(i);
        VecTypeOut vec_out;
        
        // Process VT elements
        if constexpr (VT >= 1) vec_out.x = op(vec_in.x);
        if constexpr (VT >= 2) vec_out.y = op(vec_in.y);
        if constexpr (VT >= 3) vec_out.z = op(vec_in.z);
        if constexpr (VT >= 4) vec_out.w = op(vec_in.w);
        
        // Vectorized store
        reinterpret_cast<VecTypeOut*>(output_ptr)[i / VT] = vec_out;
    }
    
    // Handle remainder elements that don't fit in complete vectors
    for (size_type i = vec_nrows + idx; i < nrows; i += stride) {
        output_ptr[i] = op(input[i]);
    }
}

template <typename Input1Itr, typename Input2Itr, typename Op>
__global__ void kernel_binary_op(
    Input1Itr input1,
    Input2Itr input2,
    mutable_column_view output,
    Op op)
{
    
    using T1 = typename std::iterator_traits<Input1Itr>::value_type;
    using T2 = typename std::iterator_traits<Input2Itr>::value_type;
    using ResType = std::remove_cv_t<std::remove_reference_t<decltype(op(cuda::std::declval<T1>(), cuda::std::declval<T2>()))>>;
    
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type stride = blockDim.x * gridDim.x;
    const size_type nrows = output.size();

    #pragma unroll
    for (size_type i = idx; i < nrows; i += stride) {
        output.data<ResType>()[i] = op(input1[i], input2[i]);
    }
}

// Unified vectorized kernel for binary operations using iterator load_vec interface
template <typename Input1Itr, typename Input2Itr, typename Op, int VT = 4>
__global__ void kernel_binary_op_vec(
    Input1Itr input1,
    Input2Itr input2,
    mutable_column_view output,
    Op op)
{
    using T1 = typename std::iterator_traits<Input1Itr>::value_type;
    using T2 = typename std::iterator_traits<Input2Itr>::value_type;
    using ResType = std::remove_cv_t<std::remove_reference_t<decltype(op(cuda::std::declval<T1>(), cuda::std::declval<T2>()))>>;
    
    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type stride = blockDim.x * gridDim.x;
    const size_type nrows = output.size();
    const size_type vec_nrows = (nrows / VT) * VT;  // Number of elements that fit in complete vectors

    using VecTypeIn1 = typename vec_type<T1, VT>::type;
    using VecTypeIn2 = typename vec_type<T2, VT>::type;
    using VecTypeOut = typename vec_type<ResType, VT>::type;
    ResType* output_ptr = output.data<ResType>();
    
    // Vectorized processing - each thread processes VT elements at a time
    #pragma unroll
    for (size_type i = idx * VT; i < vec_nrows; i += stride * VT) {
        // Vectorized loads via iterator interface
        VecTypeIn1 vec_in1 = input1.template load_vec<VT>(i);
        VecTypeIn2 vec_in2 = input2.template load_vec<VT>(i);
        VecTypeOut vec_out;
        
        // Process VT elements
        if constexpr (VT >= 1) vec_out.x = op(vec_in1.x, vec_in2.x);
        if constexpr (VT >= 2) vec_out.y = op(vec_in1.y, vec_in2.y);
        if constexpr (VT >= 3) vec_out.z = op(vec_in1.z, vec_in2.z);
        if constexpr (VT >= 4) vec_out.w = op(vec_in1.w, vec_in2.w);
        
        // Vectorized store
        reinterpret_cast<VecTypeOut*>(output_ptr)[i / VT] = vec_out;
    }
    
    // Handle remainder elements that don't fit in complete vectors
    for (size_type i = vec_nrows + idx; i < nrows; i += stride) {
        output_ptr[i] = op(input1[i], input2[i]);
    }
}

// Two iterator type parameters because Select's true/false operands
// can be a mix of column_view_itr / constant_itr (e.g. column-vs-
// scalar in `case when ... then col else 0`). Both iterators yield
// the same value type by precondition (launch_select rejects type
// mismatches).
template <typename InputItr1, typename InputItr2>
__global__ void kernel_select_op(
    InputItr1 input1,
    InputItr2 input2,
    column_view select_mask,
    mutable_column_view output)
{
    using T = typename std::iterator_traits<InputItr1>::value_type;
    using ResType = std::remove_cv_t<std::remove_reference_t<T>>;

    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type stride = blockDim.x * gridDim.x;
    const size_type nrows = output.size();

    #pragma unroll
    for (size_type i = idx; i < nrows; i += stride) {
        output.data<ResType>()[i] = select_mask.data<bool>()[i] ? input1[i] : input2[i];
    }
}

// Unified vectorized kernel for select operation using iterator load_vec interface
template <typename InputItr, int VT = 4>
__global__ void kernel_select_op_vec(
    InputItr input1,
    InputItr input2,
    column_view select_mask,
    mutable_column_view output)
{   
    using T = typename std::iterator_traits<InputItr>::value_type;
    using ResType = std::remove_cv_t<std::remove_reference_t<T>>;

    const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type stride = blockDim.x * gridDim.x;
    const size_type nrows = output.size();
    const size_type vec_nrows = (nrows / VT) * VT;  // Number of elements that fit in complete vectors

    using VecTypeIn = typename vec_type<T, VT>::type;
    using VecTypeOut = typename vec_type<ResType, VT>::type;
    const bool* mask_ptr = select_mask.data<bool>();
    ResType* output_ptr = output.data<ResType>();
    
    // Vectorized processing - each thread processes VT elements at a time
    #pragma unroll
    for (size_type i = idx * VT; i < vec_nrows; i += stride * VT) {
        // Vectorized loads via iterator interface
        VecTypeIn vec_in1 = input1.template load_vec<VT>(i);
        VecTypeIn vec_in2 = input2.template load_vec<VT>(i);
        VecTypeOut vec_out;
        
        // Process VT elements with mask selection
        if constexpr (VT >= 1) vec_out.x = mask_ptr[i + 0] ? vec_in1.x : vec_in2.x;
        if constexpr (VT >= 2) vec_out.y = mask_ptr[i + 1] ? vec_in1.y : vec_in2.y;
        if constexpr (VT >= 3) vec_out.z = mask_ptr[i + 2] ? vec_in1.z : vec_in2.z;
        if constexpr (VT >= 4) vec_out.w = mask_ptr[i + 3] ? vec_in1.w : vec_in2.w;
        
        // Vectorized store
        reinterpret_cast<VecTypeOut*>(output_ptr)[i / VT] = vec_out;
    }
    
    // Handle remainder elements that don't fit in complete vectors
    for (size_type i = vec_nrows + idx; i < nrows; i += stride) {
        output_ptr[i] = mask_ptr[i] ? input1[i] : input2[i];
    }
}

struct unary_op_launch_helper {
    template<typename Op, typename... Args>
    column_ptr launch(
        column_view input,
        cudaStream_t stream,
        Args... args)
    {
        Op op{std::forward<Args>(args)...};
        return type_dispatcher(
            input.type(), 
            *this, 
            input, 
            op, 
            stream);
    }

    template<typename Op, typename... Args>
    void launch(
        Expr const& expr,
        intermediate_storage& storage,
        cudaStream_t stream,
        Args... args)
    {
        auto operand = storage.load(expr.operand_regs[0]);
        auto result = std::visit(overloaded {
            [&](gamax::column_ptr&& col_ptr) -> gamax::column_ptr {
                return this->template launch<Op>(
                    col_ptr->view(), 
                    stream, 
                    std::forward<Args>(args)...);
            },
            [&](gamax::column_view col) -> gamax::column_ptr {
                return this->template launch<Op>(
                    col, 
                    stream,
                    std::forward<Args>(args)...);
            },
            [&](gamax::scalar s) -> gamax::column_ptr {
                throw std::runtime_error("Unary operation on scalar not allowed");
            }
        }, std::move(operand));
        storage.store(expr.return_reg, std::move(result));
    }

    template<typename T1, typename Op>
    column_ptr operator() (
        column_view input,
        Op op,
        cudaStream_t stream)
    {
        using ResType = decltype(op(cuda::std::declval<T1>()));
        auto output = gamax::make_column(
            data_type{type_to_id<ResType>()}, input.size(), stream);
        const int blockSize = 256;
        const int numBlocks = (input.size() + blockSize - 1) / blockSize;
        kernel_unary_op<<<numBlocks, blockSize, 0, stream>>>(
            make_iterator<T1>(input), output->mutable_view(), op);
        
        return output;
    }
};

struct binary_op_launch_helper {
    template<typename Op, typename Input1, typename Input2, typename... Args>
    column_ptr launch(
        Input1 input1,
        Input2 input2,
        cudaStream_t stream,
        Args... args)
    {
        Op op{std::forward<Args>(args)...};
        return double_type_dispatcher(
            input1.type(), 
            input2.type(), 
            *this, 
            input1, 
            input2, 
            op, 
            stream);
    }

    template<typename Input1, typename Input2>
    column_ptr launch_select(
        Input1 input1,
        Input2 input2,
        column_view select_mask,
        cudaStream_t stream)
    {
        if(input1.type() != input2.type()) {
            throw std::runtime_error("Input columns must have the same type for select operation");
        }

        return type_dispatcher(
            input1.type(), 
            *this, 
            input1, 
            input2, 
            select_mask, 
            stream);
    }

    template<typename Op, typename... Args>
    void launch(
        Expr const& expr,
        intermediate_storage& storage,
        cudaStream_t stream,
        Args... args)
    {
        auto o1 = storage.load(expr.operand_regs[0]);
        auto o2 = storage.load(expr.operand_regs[1]);

        auto result = std::visit(overloaded {
            [&](gamax::column_ptr&& col_ptr, gamax::column_ptr&& col_ptr2) -> gamax::column_ptr {
                return this->template launch<Op>(
                    col_ptr->view(), 
                    col_ptr2->view(),
                    stream, 
                    std::forward<Args>(args)...);
            },
            [&](column_view col, column_view col2) -> gamax::column_ptr {
                return this->template launch<Op>(
                    col, 
                    col2,
                    stream,
                    std::forward<Args>(args)...);
            },
            [&](gamax::column_ptr&& col_ptr, column_view col2) -> gamax::column_ptr {
                return this->template launch<Op>(
                    col_ptr->view(), 
                    col2,
                    stream,
                    std::forward<Args>(args)...);
            },
            [&](column_view col, gamax::column_ptr&& col_ptr2) -> gamax::column_ptr {
                return this->template launch<Op>(
                    col, 
                    col_ptr2->view(),
                    stream,
                    std::forward<Args>(args)...);
            },
            [&](gamax::column_ptr&& col_ptr, gamax::scalar s2) -> gamax::column_ptr {
                return this->template launch<Op>(
                    col_ptr->view(), 
                    s2,
                    stream,
                    std::forward<Args>(args)...);
            },
            [&](gamax::scalar s1, gamax::column_ptr&& col_ptr2) -> gamax::column_ptr {
                return this->template launch<Op>(
                    s1, 
                    col_ptr2->view(),
                    stream,
                    std::forward<Args>(args)...);
            },
            [&](column_view col, gamax::scalar s2) -> gamax::column_ptr {
                return this->template launch<Op>(
                    col, 
                    s2,
                    stream,
                    std::forward<Args>(args)...);
            },
            [&](gamax::scalar s1, column_view col2) -> gamax::column_ptr {
                return this->template launch<Op>(
                    s1, 
                    col2,
                    stream,
                    std::forward<Args>(args)...);
            },
            [&](gamax::scalar s, gamax::scalar s2) -> gamax::column_ptr {
                throw std::runtime_error("Binary operation with two scalars not allowed");
            }
        }, std::move(o1), std::move(o2));
        storage.store(expr.return_reg, std::move(result));
    }

    template<typename T1, typename T2, typename Input1, typename Input2, typename Op>
    column_ptr operator() (
        Input1 input1,
        Input2 input2,
        Op op,
        cudaStream_t stream)
    {
        using ResType = decltype(op(cuda::std::declval<T1>(), cuda::std::declval<T2>()));

        const auto output_size = get_output_size(input1, input2);

        auto output = gamax::make_column(
            data_type{type_to_id<ResType>()}, output_size, stream);
        
        constexpr int VT = 4; // Vector width for vectorized processing
        const int blockSize = 256;
        const int numBlocks = (output_size + blockSize * VT - 1) / (blockSize * VT);
        
        kernel_binary_op_vec<<<numBlocks, blockSize, 0, stream>>>(
            make_iterator<T1>(input1), 
            make_iterator<T2>(input2), 
            output->mutable_view(), 
            op);
        
        return output;
    }

    // Select-kernel overload. Single template type because
    // `launch_select` precondition guarantees the t/f operands have
    // matching dtypes; type_dispatcher dispatches that one type and
    // both make_iterator calls share it.
    template<typename T, typename Input1, typename Input2>
    column_ptr operator() (
        Input1 input1,
        Input2 input2,
        column_view select_mask,
        cudaStream_t stream)
    {
        const auto output_size = get_output_size(input1, input2);
        auto output = gamax::make_column(input1.type(), output_size, stream);

        const int blockSize = 256;
        const int numBlocks = (output_size + blockSize - 1) / blockSize;

        kernel_select_op<<<numBlocks, blockSize, 0, stream>>>(
            make_iterator<T>(input1),
            make_iterator<T>(input2),
            select_mask,
            output->mutable_view());

        return output;
    }

private:
    gamax::size_type get_output_size(
        column_view input1,
        column_view input2)
    {
        if(input1.size() != input2.size()) {
            throw std::runtime_error("Input columns must have the same size");
        }
        return input1.size();
    }

    gamax::size_type get_output_size(
        column_view input1,
        gamax::scalar input2)
    {
        return input1.size();
    }

    gamax::size_type get_output_size(
        gamax::scalar input1,
        column_view input2)
    {
        return input2.size();
    }
};

// Broadcast a scalar value into a fresh column of `n` rows. Used to
// materialise an expression program whose final register is a scalar
// (e.g. a project item that is a literal constant) — the public
// contract of `evaluate_expr_nofuse` is to return a column.
struct broadcast_scalar_helper {
    template <typename T,
              std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    gamax::column_ptr operator()(gamax::scalar const& s,
                                 gamax::size_type n,
                                 cudaStream_t stream) const
    {
        auto col = gamax::make_column<T>(
            n, stream, gamax::buffer_init::uninitialized);
        T val = s.get<T>();
        // Green-safe: thrust::fill_n routes to cub::DeviceTransform, which fails
        // on green-context streams. device_fill is a plain-kernel replacement.
        gamax::util::device::device_fill(
            col->mutable_view().template data<T>(), n, val, stream);
        return col;
    }

    template <typename T,
              std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
    gamax::column_ptr operator()(gamax::scalar const&,
                                 gamax::size_type,
                                 cudaStream_t) const
    {
        throw std::runtime_error(
            "evaluate_expr_nofuse: scalar broadcast for non-arithmetic "
            "types not supported");
    }
};

gamax::column_ptr broadcast_scalar(gamax::scalar const& s,
                                   gamax::size_type n,
                                   cudaStream_t stream)
{
    if (s.type().id() == gamax::type_id::STRING) {
        throw std::runtime_error(
            "evaluate_expr_nofuse: STRING scalar broadcast not supported");
    }
    return gamax::type_dispatcher(
        s.type(), broadcast_scalar_helper{}, s, n, stream);
}

// Borrow a `column_view` from a Select-condition operand. The cond
// variant is held by the caller's scope while the kernel runs, so
// the returned view is valid for the kernel's lifetime.
column_view cond_view_of(intermediate_storage::operand_type& v)
{
    return std::visit(overloaded {
        [](gamax::column_ptr& cp) -> column_view { return cp->view(); },
        [](column_view cv)         -> column_view { return cv; },
        [](gamax::scalar) -> column_view {
            throw std::runtime_error(
                "evaluate_expr_nofuse: Select condition must be a "
                "column, not a scalar");
        },
    }, v);
}

void evaluate_one_expr(
    Expr const& expr,
    gamax::table_view& table,
    intermediate_storage& storage,
    cudaStream_t stream)
{
    using namespace gamax::expression::detail;
    switch(expr.id) {
        case ExprId::Constant: {
            storage.store(
                expr.return_reg, 
                expr.data.c.value);
            break;
        }
        case ExprId::Column: {
            storage.store(
                expr.return_reg, 
                table.get_column(expr.data.col.cid));
            break;
        }
        case ExprId::Negate: {
            unary_op_launch_helper launcher;
            launcher.launch<negate_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::LeftShift: {
            unary_op_launch_helper launcher;
            launcher.launch<left_shift_helper_v2>(
                expr,
                storage, 
                stream, 
                expr.data.lshift.bits);
            break;
        }
        case ExprId::RightShift: {
            unary_op_launch_helper launcher;
            launcher.launch<right_shift_helper_v2>(
                expr,
                storage, 
                stream, 
                expr.data.rshift.bits);
            break;
        }
        case ExprId::Add: {
            binary_op_launch_helper launcher;
            launcher.launch<add_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Substract: {
            binary_op_launch_helper launcher;
            launcher.launch<substract_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Multiply: {
            binary_op_launch_helper launcher;
            launcher.launch<multiply_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Divide: {
            binary_op_launch_helper launcher;
            launcher.launch<divide_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Sqrt: {
            unary_op_launch_helper launcher;
            launcher.launch<sqrt_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Log: {
            unary_op_launch_helper launcher;
            launcher.launch<log_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Exp: {
            unary_op_launch_helper launcher;
            launcher.launch<exp_helper>(
                expr,
                storage,
                stream);
            break;
        }
        case ExprId::YearOf: {
            unary_op_launch_helper launcher;
            launcher.launch<year_of_helper>(
                expr,
                storage,
                stream);
            break;
        }
        case ExprId::GreaterThan: {
            binary_op_launch_helper launcher;
            launcher.launch<gt_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::LessThan: {
            binary_op_launch_helper launcher;
            launcher.launch<lt_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::GreaterEqual: {
            binary_op_launch_helper launcher;
            launcher.launch<ge_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::LessEqual: {
            binary_op_launch_helper launcher;
            launcher.launch<le_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Equal: {
            binary_op_launch_helper launcher;
            launcher.launch<eq_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::And: {
            binary_op_launch_helper launcher;
            launcher.launch<and_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Or: {
            binary_op_launch_helper launcher;
            launcher.launch<or_helper>(
                expr,
                storage, 
                stream);
            break;
        }
        case ExprId::Select: {
            // Select(cond, t, f): elementwise `cond ? t : f`. The
            // existing `binary_op_launch_helper::launch_select` plus
            // `kernel_select_op` infrastructure already implements
            // the kernel; this case wires the variant dispatch over
            // the t/f operand types and supplies cond as the mask.
            //
            // The cond operand must be column-shaped (BOOL8) — it
            // typically arrives as a `column_ptr` from a prior
            // comparison op.
            auto cond_v = storage.load(expr.operand_regs[0]);
            auto t_v    = storage.load(expr.operand_regs[1]);
            auto f_v    = storage.load(expr.operand_regs[2]);
            column_view cond_view = cond_view_of(cond_v);

            binary_op_launch_helper launcher;
            auto result = std::visit(overloaded {
                [&](gamax::column_ptr&& tp, gamax::column_ptr&& fp)
                    -> gamax::column_ptr {
                    return launcher.launch_select(
                        tp->view(), fp->view(), cond_view, stream);
                },
                [&](column_view tv, column_view fv)
                    -> gamax::column_ptr {
                    return launcher.launch_select(
                        tv, fv, cond_view, stream);
                },
                [&](gamax::column_ptr&& tp, column_view fv)
                    -> gamax::column_ptr {
                    return launcher.launch_select(
                        tp->view(), fv, cond_view, stream);
                },
                [&](column_view tv, gamax::column_ptr&& fp)
                    -> gamax::column_ptr {
                    return launcher.launch_select(
                        tv, fp->view(), cond_view, stream);
                },
                [&](gamax::column_ptr&& tp, gamax::scalar fs)
                    -> gamax::column_ptr {
                    return launcher.launch_select(
                        tp->view(), fs, cond_view, stream);
                },
                [&](gamax::scalar ts, gamax::column_ptr&& fp)
                    -> gamax::column_ptr {
                    return launcher.launch_select(
                        ts, fp->view(), cond_view, stream);
                },
                [&](column_view tv, gamax::scalar fs)
                    -> gamax::column_ptr {
                    return launcher.launch_select(
                        tv, fs, cond_view, stream);
                },
                [&](gamax::scalar ts, column_view fv)
                    -> gamax::column_ptr {
                    return launcher.launch_select(
                        ts, fv, cond_view, stream);
                },
                [&](gamax::scalar ts, gamax::scalar fs)
                    -> gamax::column_ptr {
                    // Both branches are scalars (e.g. `case when ...
                    // then 1 else 0`). Broadcast the true branch to
                    // a column of cond's row count, then dispatch
                    // via the existing column-vs-scalar select path.
                    auto t_col = broadcast_scalar(
                        ts, cond_view.size(), stream);
                    return launcher.launch_select(
                        t_col->view(), fs, cond_view, stream);
                },
            }, std::move(t_v), std::move(f_v));
            storage.store(expr.return_reg, std::move(result));
            break;
        }
        default:
            throw std::runtime_error("Unsupported expression type");
    }
}
} // namespace anonymous
    
std::unique_ptr<gamax::column> evaluate_expr_nofuse(
    gamax::table_view table,
    ExprTree& et,
    cudaStream_t stream)
{
    intermediate_storage storage;

    auto program = et.make_program_host();
    for(const auto& expr : program) {
        evaluate_one_expr(expr, table, storage, stream);
    }

    // Materialise whatever the final register holds into an owned
    // column. The intermediate-storage variant carries three forms;
    // any of them is a valid program output:
    //   * column_ptr  — a freshly-produced column from an op kernel.
    //                   Move it out as-is.
    //   * column_view — a borrowed slice of the input table. Happens
    //                   when the expression's root is `Column` (e.g.
    //                   a project item that turns out to be a single
    //                   column reference and slipped past phys's
    //                   `is_column_ref()` early-out). Deep-copy so
    //                   the caller owns the result.
    //   * scalar      — a literal constant. Happens when a project
    //                   item is `lit(0)` etc. Broadcast to a column
    //                   of `table.num_rows()` rows so the result has
    //                   the right shape.
    auto result_variant = storage.load(program.back().return_reg);
    return std::visit(overloaded {
        [](gamax::column_ptr&& p) -> gamax::column_ptr {
            return std::move(p);
        },
        [&](column_view v) -> gamax::column_ptr {
            return gamax::copy_column(v, stream);
        },
        [&](gamax::scalar s) -> gamax::column_ptr {
            return broadcast_scalar(s, table.num_rows(), stream);
        },
    }, std::move(result_variant));
}
} // namespace gamax::expression
