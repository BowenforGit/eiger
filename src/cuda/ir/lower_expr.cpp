#include "ir/lower_expr.hpp"

#include <stdexcept>
#include <string>
#include <variant>

#include "memory/scalar.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"

namespace gamax::ir {

namespace {

[[noreturn]] void lower_error(const std::string& what) {
    throw std::runtime_error("ir::lower_expr: " + what);
}

gamax::scalar scalar_from_literal(const LiteralValue& v, gamax::data_type type) {
    using gamax::type_id;
    return std::visit([type](auto&& x) -> gamax::scalar {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, int64_t>) {
            // Narrow int64 literal to the Expr's declared type so the
            // scalar's stored type matches what the ExprTree consumer
            // expects.
            switch (type.id()) {
                case type_id::INT8:  return gamax::scalar(static_cast<int8_t>(x));
                case type_id::INT16: return gamax::scalar(static_cast<int16_t>(x));
                case type_id::INT32: return gamax::scalar(static_cast<int32_t>(x));
                case type_id::INT64: return gamax::scalar(x);
                case type_id::UINT8:  return gamax::scalar(static_cast<uint8_t>(x));
                case type_id::UINT16: return gamax::scalar(static_cast<uint16_t>(x));
                case type_id::UINT32: return gamax::scalar(static_cast<uint32_t>(x));
                case type_id::UINT64: return gamax::scalar(static_cast<uint64_t>(x));
                case type_id::FLOAT32: return gamax::scalar(static_cast<float>(x));
                case type_id::FLOAT64: return gamax::scalar(static_cast<double>(x));
                default: lower_error("integer literal cannot be cast to target scalar type");
            }
        } else if constexpr (std::is_same_v<T, double>) {
            switch (type.id()) {
                case type_id::FLOAT32: return gamax::scalar(static_cast<float>(x));
                case type_id::FLOAT64: return gamax::scalar(x);
                default: lower_error("floating literal cannot be cast to non-floating scalar");
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            return gamax::scalar(x);
        } else if constexpr (std::is_same_v<T, std::string>) {
            (void)type;
            lower_error("string literal has no gamax::scalar form — "
                        "use a StringOp in a Predicate context");
        }
    }, v);
}

using gamax::expression::ExprTree;
using gamax::select::Predicate;

ExprTree lower_bin_arith(const Expr& e, const Schema& input) {
    auto l = lower_expr_to_tree(e.children[0], input);
    auto r = lower_expr_to_tree(e.children[1], input);
    switch (e.arith_op) {
        case ArithOp::Add: return ExprTree::add(std::move(l), std::move(r));
        case ArithOp::Sub: return ExprTree::subtract(std::move(l), std::move(r));
        case ArithOp::Mul: return ExprTree::multiply(std::move(l), std::move(r));
        case ArithOp::Div: return ExprTree::divide(std::move(l), std::move(r));
    }
    lower_error("unknown ArithOp");
}

ExprTree lower_bin_compare_to_tree(const Expr& e, const Schema& input) {
    auto l = lower_expr_to_tree(e.children[0], input);
    auto r = lower_expr_to_tree(e.children[1], input);
    switch (e.compare_op) {
        case CompareOp::Eq: return ExprTree::equal(std::move(l), std::move(r));
        case CompareOp::Lt: return ExprTree::less_than(std::move(l), std::move(r));
        case CompareOp::Le: return ExprTree::less_equal(std::move(l), std::move(r));
        case CompareOp::Gt: return ExprTree::greater_than(std::move(l), std::move(r));
        case CompareOp::Ge: return ExprTree::greater_equal(std::move(l), std::move(r));
        case CompareOp::Ne:
            lower_error("CompareOp::Ne has no direct ExprTree form; use "
                        "lower_expr_to_predicate (it emits NOT(==))");
    }
    lower_error("unknown CompareOp");
}

ExprTree lower_bin_logical_to_tree(const Expr& e, const Schema& input) {
    auto l = lower_expr_to_tree(e.children[0], input);
    auto r = lower_expr_to_tree(e.children[1], input);
    switch (e.logical_op) {
        case LogicalOp::And:
            return ExprTree::logical_and(std::move(l), std::move(r));
        case LogicalOp::Or:
            return ExprTree::logical_or(std::move(l), std::move(r));
    }
    lower_error("unknown LogicalOp");
}

std::shared_ptr<gamax::string_scalar> make_shared_string_scalar(
    const std::string& s, cudaStream_t stream)
{
    // make_string_scalar returns unique_ptr; Predicate's string leaves
    // want shared_ptr. Convert explicitly.
    return std::shared_ptr<gamax::string_scalar>(
        gamax::make_string_scalar(s, stream).release());
}

std::vector<std::shared_ptr<gamax::string_scalar>> make_string_scalar_vec(
    const std::vector<std::string>& ss, cudaStream_t stream)
{
    std::vector<std::shared_ptr<gamax::string_scalar>> out;
    out.reserve(ss.size());
    for (const auto& s : ss) out.push_back(make_shared_string_scalar(s, stream));
    return out;
}

Predicate lower_string_op_leaf(
    const Expr& e, const Schema& input, cudaStream_t stream)
{
    if (e.children.size() != 1 || e.children[0].kind != ExprKind::ColRef) {
        lower_error("StringOp must wrap a ColRef column");
    }
    auto cid = resolve_ordinal(input, e.children[0].col_ref);
    switch (e.string_op) {
        case StringOpKind::Equal:
            return Predicate::string_equal(
                cid, make_shared_string_scalar(e.string_values.front(), stream));
        case StringOpKind::Startswith:
            return Predicate::string_startswith(
                cid, make_shared_string_scalar(e.string_values.front(), stream));
        case StringOpKind::Endswith:
            return Predicate::string_endswith(
                cid, make_shared_string_scalar(e.string_values.front(), stream));
        case StringOpKind::Contains:
            return Predicate::string_contains(
                cid, make_shared_string_scalar(e.string_values.front(), stream));
        case StringOpKind::ContainsAny:
            return Predicate::string_contains_multiple(
                cid, make_string_scalar_vec(e.string_values, stream));
        case StringOpKind::In:
            return Predicate::string_in(
                cid, make_string_scalar_vec(e.string_values, stream));
    }
    lower_error("unknown StringOpKind");
}

} // namespace

gamax::size_type resolve_ordinal(const Schema& input, ColumnId id) {
    auto i = input.index_of(id);
    if (!i) {
        lower_error("ColumnId " + std::to_string(id.value) +
                    " not in input schema");
    }
    return static_cast<gamax::size_type>(*i);
}

bool expr_contains_string_op(const Expr& e) {
    if (e.kind == ExprKind::StringOp) return true;
    for (const auto& c : e.children) {
        if (expr_contains_string_op(c)) return true;
    }
    return false;
}

ExprTree lower_expr_to_tree(const Expr& e, const Schema& input) {
    switch (e.kind) {
        case ExprKind::ColRef:
            return ExprTree::column(resolve_ordinal(input, e.col_ref), e.type);
        case ExprKind::Literal:
            return ExprTree::constant(scalar_from_literal(e.literal, e.type));
        case ExprKind::BinArith:
            return lower_bin_arith(e, input);
        case ExprKind::BinCompare:
            return lower_bin_compare_to_tree(e, input);
        case ExprKind::BinLogical:
            return lower_bin_logical_to_tree(e, input);
        case ExprKind::Unary:
            // Only Unary::Not exists today; ExprTree has no NOT.
            lower_error("Unary NOT cannot be lowered to ExprTree — "
                        "wrap the surrounding boolean context in a Predicate");
        case ExprKind::Case:
            return ExprTree::select(
                lower_expr_to_tree(e.children[0], input),
                lower_expr_to_tree(e.children[1], input),
                lower_expr_to_tree(e.children[2], input));
        case ExprKind::StringOp:
            lower_error("StringOp cannot appear inside a computational "
                        "ExprTree; materialize it as a bool column first");
    }
    lower_error("unknown ExprKind");
}

Predicate lower_expr_to_predicate(
    const Expr& e, const Schema& input, cudaStream_t stream)
{
    // If the subtree is purely numeric/logical/comparison with no
    // StringOps, fold it into a single Predicate::expr wrapping one
    // ExprTree — minimizes kernel launches. Ne is decomposed here to
    // stay numeric-only.
    if (!expr_contains_string_op(e)) {
        // Handle Ne specifically, because ExprTree doesn't natively
        // support !=, so we rewrite to NOT(==) at the Predicate level.
        if (e.kind == ExprKind::BinCompare && e.compare_op == CompareOp::Ne) {
            auto eq = ExprTree::equal(
                lower_expr_to_tree(e.children[0], input),
                lower_expr_to_tree(e.children[1], input));
            return Predicate::logical_not(Predicate::expr(std::move(eq)));
        }
        // For Unary::Not over a non-BOOL subtree we also can't produce
        // a single ExprTree (no NOT node there), so decompose.
        if (e.kind == ExprKind::Unary && e.unary_op == UnaryOp::Not) {
            return Predicate::logical_not(
                lower_expr_to_predicate(e.children[0], input, stream));
        }
        return Predicate::expr(lower_expr_to_tree(e, input));
    }

    // Contains at least one StringOp somewhere. Stay at the Predicate
    // level for combinators and emit string leaves directly.
    switch (e.kind) {
        case ExprKind::BinLogical:
            switch (e.logical_op) {
                case LogicalOp::And:
                    return Predicate::logical_and(
                        lower_expr_to_predicate(e.children[0], input, stream),
                        lower_expr_to_predicate(e.children[1], input, stream));
                case LogicalOp::Or:
                    return Predicate::logical_or(
                        lower_expr_to_predicate(e.children[0], input, stream),
                        lower_expr_to_predicate(e.children[1], input, stream));
            }
            lower_error("unknown LogicalOp");
        case ExprKind::Unary:
            return Predicate::logical_not(
                lower_expr_to_predicate(e.children[0], input, stream));
        case ExprKind::StringOp:
            return lower_string_op_leaf(e, input, stream);
        default:
            // A purely numeric subtree nested under a mixed predicate:
            // fold it into a single Predicate::expr wrapping one tree.
            return Predicate::expr(lower_expr_to_tree(e, input));
    }
}

} // namespace gamax::ir
