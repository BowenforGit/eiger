#include <memory>
#include <vector>
#include <variant>
#include <unordered_map>
#include <type_traits>
#include <cuda/std/utility>
#include <thrust/iterator/constant_iterator.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "memory/device_buffer.hpp"
#include "expr/expr_fast.hpp"
#include "expr/expr_impl.cuh"
#include "memory/scalar.hpp"
#include "utils/utils.cuh"
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

template <typename InputItr>
__global__ void kernel_select_op(
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
        auto output = gamax::make_column(data_type{type_to_id<ResType>()}, input.size());
        const int blockSize = 256;
        const int numBlocks = (input.size() + blockSize - 1) / blockSize;
        kernel_unary_op<<<numBlocks, blockSize, 0, stream>>>(
            make_iterator<T1>(input), output->mutable_view(), op);
        
        // Ensure the kernel is finished before returning otherwise
        // the unique_ptr may go out of scope before the kernel is done
        cudaStreamSynchronize(stream);
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

        auto output = gamax::make_column(data_type{type_to_id<ResType>()}, output_size);
        
        constexpr int VT = 4; // Vector width for vectorized processing
        const int blockSize = 256;
        const int numBlocks = (output_size + blockSize * VT - 1) / (blockSize * VT);
        
        kernel_binary_op_vec<<<numBlocks, blockSize, 0, stream>>>(
            make_iterator<T1>(input1), 
            make_iterator<T2>(input2), 
            output->mutable_view(), 
            op);
        
        // Ensure the kernel is finished before returning otherwise
        // the unique_ptr may go out of scope before the kernel is done
        cudaStreamSynchronize(stream);
        return output;
    }

    template<typename T1, typename T2, typename Input1, typename Input2>
    column_ptr operator() (
        Input1 input1,
        Input2 input2,
        column_view select_mask,
        cudaStream_t stream)
    {
        const auto output_size = get_output_size(input1, input2);
        auto output = gamax::make_column(input1.type(), output_size);
        
        const int blockSize = 256;
        const int numBlocks = (output_size + blockSize - 1) / blockSize;
        
        kernel_select_op<<<numBlocks, blockSize, 0, stream>>>(
            make_iterator<T1>(input1), 
            make_iterator<T2>(input2), 
            select_mask,
            output->mutable_view());
        
        // Ensure the kernel is finished before returning otherwise
        // the unique_ptr may go out of scope before the kernel is done
        cudaStreamSynchronize(stream);
        
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
        case ExprId::Select: 
         // skip for now
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

    auto result_variant = storage.load(program.back().return_reg);
    
    if(!std::holds_alternative<gamax::column_ptr>(result_variant)) {
        throw std::runtime_error("The final result of the expression is not a column");
    }
    return std::get<gamax::column_ptr>(std::move(result_variant));
}
} // namespace gamax::expression