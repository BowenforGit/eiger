#pragma once

#include <memory>
#include <string>
#include <vector>

#include "expr/expr_fast.hpp"
#include "memory/scalar.hpp"
#include "types/data_type.hpp"

namespace gamax::expression {

enum class AstNodeKind {
    Literal,
    ColumnRef,
    Unary,
    Binary,
    Select,
    ShiftLeft,
    ShiftRight
};

enum class AstUnaryOp {
    Negate,
    Sqrt,
    Log,
    Exp
};

enum class AstBinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
    Equal,
    And,
    Or
};

struct AstNode {
    AstNodeKind kind {AstNodeKind::Literal};
    gamax::data_type result_type {};
    gamax::scalar literal_value {};
    gamax::size_type column_index {0};
    AstUnaryOp unary_op {AstUnaryOp::Negate};
    AstBinaryOp binary_op {AstBinaryOp::Add};
    int shift_bits {0};
    std::unique_ptr<AstNode> lhs;
    std::unique_ptr<AstNode> rhs;
    std::unique_ptr<AstNode> third;

    AstNode() = default;
    AstNode(AstNode&&) noexcept = default;
    AstNode& operator=(AstNode&&) noexcept = default;
    AstNode(const AstNode&) = delete;
    AstNode& operator=(const AstNode&) = delete;
};

// Parse a string expression into an AST using a lightweight lexer/parser tailored to the
// supported operators in ExprTree. Column references are written as "col<index>".
AstNode parse_expression_to_ast(
    const std::string& expr,
    const std::vector<gamax::data_type>& column_types);

// Convert an AST into the existing ExprTree structure.
ExprTree ast_to_expr_tree(const AstNode& root);

// Convenience: parse a string into an ExprTree in one step.
ExprTree parse_expression_to_tree(
    const std::string& expr,
    const std::vector<gamax::data_type>& column_types);

} // namespace gamax::expression
