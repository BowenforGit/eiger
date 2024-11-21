#pragma once

#include <memory>
#include <cuda.h>
#include <variant>

#include "memory/column.hpp"
#include "expr/scalar.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax::expression {
enum ExprId {
    Constant,
    Column,
    Negate,
    LeftShift,
    Add,
    Substract,
    Multiply
};

class Expr {
public:
    Expr() = default;
    Expr(ExprId id, gamax::data_type type) : _id(id), _type(type) {}
    __host__ __device__ ExprId id() const noexcept { return _id; }
    __host__ __device__ gamax::data_type type() const noexcept { return _type; }
private:
    ExprId _id;
    gamax::data_type _type;
};

class ConstantExpr : public Expr {
public:
    ConstantExpr(gamax::scalar value) : Expr(Constant, value.type()), _value(value) {}

    __host__ __device__ gamax::scalar eval() const noexcept {
        return _value;
    }

private:
    gamax::scalar _value;
};

class ColumnExpr : public Expr {
    struct column_access_helper {
        template <typename T>
        __host__ __device__ gamax::scalar operator()(gamax::column_view col, gamax::size_type i) const noexcept {
            return gamax::scalar(col.data<T>()[i]);
        }
    };

public:
    ColumnExpr(gamax::column_view column) : Expr(Column, column.type()), _column(column) {}

    __host__ __device__ gamax::scalar eval(gamax::size_type i) const noexcept {
        return gamax::type_dispatcher(_column.type(), column_access_helper{}, _column, i);
    }

    __host__ __device__ gamax::size_type nrows() const noexcept {
        return _column.size();
    }

private:
    gamax::column_view _column;
};

class NegateExpr : public Expr {
    struct negate_helper {
        template <typename T>
        __host__ __device__ gamax::scalar operator()(gamax::scalar value) const {
            return gamax::scalar(-value.get<T>());
        }
    };

public:
    NegateExpr(gamax::data_type dtype) : Expr(Negate, dtype) {}

    __host__ __device__ gamax::scalar eval(gamax::scalar s) const {
        return gamax::type_dispatcher(s.type(), negate_helper{}, s);
    }
};

class LeftShiftExpr : public Expr {
    struct right_shift_helper {
        template <typename T>
        __host__ __device__ gamax::scalar operator()(gamax::scalar value, int bits) const {
            return gamax::scalar(value.get<T>() << bits);
        }
    };

public:
    LeftShiftExpr(gamax::data_type dtype, int bits) : Expr(LeftShift, dtype), _bits(bits) {}

    __host__ __device__ gamax::scalar eval(gamax::scalar s) const {
        return gamax::type_dispatcher(s.type(), right_shift_helper {}, s, _bits);
    }

private:
    int _bits;
};

class AddExpr : public Expr {
struct add_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(e1.get<T>() + e2.get<U>());
    }
};

public:
    AddExpr(gamax::data_type dtype) : Expr(Add, dtype) {}

    __host__ __device__ gamax::scalar eval(gamax::scalar s1, gamax::scalar s2) const {
        return gamax::double_type_dispatcher(s1.type(), s2.type(), add_helper{}, s1, s2);
    }
};

class SubstractExpr : public Expr {
struct substract_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(e1.get<T>() - e2.get<U>());
    }
};

public:
    SubstractExpr(gamax::data_type dtype) : Expr(Substract, dtype) {}

    __host__ __device__ gamax::scalar eval(gamax::scalar s1, gamax::scalar s2) const {
        return gamax::double_type_dispatcher(s1.type(), s2.type(), substract_helper {}, s1, s2);
    }
};

class MultiplyExpr : public Expr {
struct mult_helper {
    template <typename T, typename U>
    __host__ __device__ gamax::scalar operator()(gamax::scalar e1, gamax::scalar e2) const {
        return gamax::scalar(e1.get<T>() * e2.get<U>());
    }
};

public:
    MultiplyExpr(gamax::data_type dtype) : Expr(Multiply, dtype) {}

    __host__ __device__ gamax::scalar eval(gamax::scalar s1, gamax::scalar s2) const {
        return gamax::double_type_dispatcher(s1.type(), s2.type(), mult_helper {}, s1, s2);
    }
};

using ExprType = std::variant<
    ConstantExpr, 
    ColumnExpr, 
    NegateExpr, 
    LeftShiftExpr, 
    AddExpr, 
    SubstractExpr, 
    MultiplyExpr
>;

struct ExprNode {
    ExprType expr;
    gamax::size_type n_children;
    gamax::size_type children[2];

    ExprNode(ExprType e) 
        : expr(e), n_children(0) 
    {
        children[0] = -1;
        children[1] = -1;
    }

    ExprNode(ExprType e, gamax::size_type child_id) 
        : expr(e), n_children(1)
    {
        children[0] = child_id;
        children[1] = -1;
    }

    ExprNode(ExprType e, gamax::size_type child_id1, gamax::size_type child_id2) 
        : expr(e), n_children(2) 
    {
        children[0] = child_id1;
        children[1] = child_id2;
    }
};

class ExprTreeDevice {
public:
    ExprTreeDevice() = default;

    ExprTreeDevice(
        std::vector<ExprNode> const& nodes, 
        gamax::size_type root_id, 
        gamax::size_type nrows, 
        cudaStream_t stream = cudaStreamDefault) 
        : _size(nodes.size()), _root_id(root_id), _nrows(nrows)
    {
        allocate_mem(&_nodes, false, _size*sizeof(ExprNode), stream);

        for (gamax::size_type i = 0; i < _size; ++i) {
            cudaMemcpyAsync(&_nodes[i], &nodes[i], sizeof(ExprNode), cudaMemcpyHostToDevice, stream);
        }
    }

    __host__ __device__ ExprNode& operator[](gamax::size_type i) {
        return _nodes[i];
    }

    __host__ __device__ const ExprNode& operator[](gamax::size_type i) const {
        return _nodes[i];
    }

    __host__ __device__ gamax::size_type size() const noexcept {
        return _size;
    }

    __host__ __device__ ExprNode& root() {
        return _nodes[_root_id];
    }

    __host__ __device__ const ExprNode& root() const {
        return _nodes[_root_id];
    }

    __host__ __device__ gamax::size_type root_id() const noexcept {
        return _root_id;
    }

    __host__ void destroy() {
        release_mem(_nodes);
    }

private:
    ExprNode* _nodes;
    gamax::size_type _size;
    gamax::size_type _root_id;
    gamax::size_type _nrows;
};

class ExprTree {
public:
    ExprTree() = default;

    gamax::size_type add_node(ExprType e, bool as_root = false) {
        _nodes.emplace_back(e);
        if(as_root) {
            _root_id = _nodes.size() - 1;
        }
        if(std::holds_alternative<ColumnExpr>(e)) {
            if(_nrows == 0) {
                _nrows = std::get<ColumnExpr>(e).nrows();
            } else if(_nrows != std::get<ColumnExpr>(e).nrows()) {
                throw std::runtime_error("ColumnExpr nodes must have the same number of rows");
            }
        }
        return _nodes.size() - 1;
    }

    gamax::size_type add_node(ExprType e, gamax::size_type child_id, bool as_root = false) {
        _nodes.emplace_back(e, child_id);
        if(as_root) {
            _root_id = _nodes.size() - 1;
        }
        return _nodes.size() - 1;
    }

    gamax::size_type add_node(ExprType e, gamax::size_type child_id1, gamax::size_type child_id2, bool as_root = false) {
        _nodes.emplace_back(e, child_id1, child_id2);
        if(as_root) {
            _root_id = _nodes.size() - 1;
        }
        return _nodes.size() - 1;
    }

    ExprNode& operator[](gamax::size_type i) {
        return _nodes[i];
    }

    const ExprNode& operator[](gamax::size_type i) const {
        return _nodes[i];
    }

    gamax::size_type size() const noexcept {
        return _nodes.size();
    }

    gamax::size_type nrows() const noexcept {
        return _nrows;
    }

    ExprNode& root() {
        return _nodes[_root_id];
    }

    const ExprNode& root() const {
        return _nodes[_root_id];
    }

    auto to_device(cudaStream_t stream = cudaStreamDefault) {
        auto deleter = [](ExprTreeDevice* t) { t->destroy(); };
        return std::unique_ptr<ExprTreeDevice, decltype(deleter)>{
            new ExprTreeDevice(_nodes, _root_id, _nrows, stream), deleter};
    }

private:
    gamax::size_type _root_id;
    std::vector<ExprNode> _nodes;
    gamax::size_type _nrows {0};
};

gamax::data_type get_expr_type(ExprType const& e);

std::unique_ptr<gamax::column> evaluate_expr(
    ExprTree& et, 
    cudaStream_t stream = cudaStreamDefault);
} // namespace gamax