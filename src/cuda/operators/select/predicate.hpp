#pragma once

#include <memory>
#include <string>
#include <vector>

#include "expr/expr_fast.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "operators/select/select.hpp"

namespace gamax::select {

enum class PredicateNodeType {
    // Leaf: numeric expression tree
    EXPR,
    // Leaf: string predicates
    STRING_EQUAL,
    STRING_STARTSWITH,
    STRING_ENDSWITH,
    STRING_CONTAINS,
    STRING_CONTAINS_MULTIPLE,
    STRING_IN,
    // Internal: logical combinators
    AND,
    OR,
    NOT
};

class Predicate {
public:
    Predicate() = default;
    Predicate(const Predicate&) = default;
    Predicate(Predicate&&) = default;
    Predicate& operator=(const Predicate&) = default;
    Predicate& operator=(Predicate&&) = default;

    // ── Leaf factories: numeric expression ──

    static Predicate expr(
        gamax::expression::ExprTree tree,
        select_options opts = {})
    {
        Predicate p;
        p._type = PredicateNodeType::EXPR;
        p._expr = std::move(tree);
        p._select_opts = opts;
        return p;
    }

    // ── Leaf factories: single-string predicates ──

    static Predicate string_equal(
        gamax::size_type cid,
        std::shared_ptr<gamax::string_scalar> value,
        gamax::string_op_options opts = {},
        gamax::string_impl_id impl = gamax::ARROW_STRING)
    {
        Predicate p;
        p._type = PredicateNodeType::STRING_EQUAL;
        p._string_cid = cid;
        p._string_value = std::move(value);
        p._string_opts = opts;
        p._string_impl = impl;
        return p;
    }

    static Predicate string_startswith(
        gamax::size_type cid,
        std::shared_ptr<gamax::string_scalar> value,
        gamax::string_op_options opts = {},
        gamax::string_impl_id impl = gamax::ARROW_STRING)
    {
        Predicate p;
        p._type = PredicateNodeType::STRING_STARTSWITH;
        p._string_cid = cid;
        p._string_value = std::move(value);
        p._string_opts = opts;
        p._string_impl = impl;
        return p;
    }

    static Predicate string_endswith(
        gamax::size_type cid,
        std::shared_ptr<gamax::string_scalar> value,
        gamax::string_op_options opts = {},
        gamax::string_impl_id impl = gamax::ARROW_STRING)
    {
        Predicate p;
        p._type = PredicateNodeType::STRING_ENDSWITH;
        p._string_cid = cid;
        p._string_value = std::move(value);
        p._string_opts = opts;
        p._string_impl = impl;
        return p;
    }

    static Predicate string_contains(
        gamax::size_type cid,
        std::shared_ptr<gamax::string_scalar> value,
        gamax::string_op_options opts = {},
        gamax::string_impl_id impl = gamax::ARROW_STRING)
    {
        Predicate p;
        p._type = PredicateNodeType::STRING_CONTAINS;
        p._string_cid = cid;
        p._string_value = std::move(value);
        p._string_opts = opts;
        p._string_impl = impl;
        return p;
    }

    // ── Leaf factories: multi-string predicates ──

    static Predicate string_contains_multiple(
        gamax::size_type cid,
        std::vector<std::shared_ptr<gamax::string_scalar>> values,
        gamax::string_op_options opts = {},
        gamax::string_impl_id impl = gamax::ARROW_STRING)
    {
        Predicate p;
        p._type = PredicateNodeType::STRING_CONTAINS_MULTIPLE;
        p._string_cid = cid;
        p._string_values = std::move(values);
        p._string_opts = opts;
        p._string_impl = impl;
        return p;
    }

    static Predicate string_in(
        gamax::size_type cid,
        std::vector<std::shared_ptr<gamax::string_scalar>> values,
        gamax::string_op_options opts = {},
        gamax::string_impl_id impl = gamax::ARROW_STRING)
    {
        Predicate p;
        p._type = PredicateNodeType::STRING_IN;
        p._string_cid = cid;
        p._string_values = std::move(values);
        p._string_opts = opts;
        p._string_impl = impl;
        return p;
    }

    // ── Internal (logical) factories ──

    static Predicate logical_and(Predicate left, Predicate right) {
        Predicate p;
        p._type = PredicateNodeType::AND;
        p._children.push_back(std::make_shared<Predicate>(std::move(left)));
        p._children.push_back(std::make_shared<Predicate>(std::move(right)));
        return p;
    }

    static Predicate logical_or(Predicate left, Predicate right) {
        Predicate p;
        p._type = PredicateNodeType::OR;
        p._children.push_back(std::make_shared<Predicate>(std::move(left)));
        p._children.push_back(std::make_shared<Predicate>(std::move(right)));
        return p;
    }

    static Predicate logical_not(Predicate child) {
        Predicate p;
        p._type = PredicateNodeType::NOT;
        p._children.push_back(std::make_shared<Predicate>(std::move(child)));
        return p;
    }

    // ── Operator overloads for convenient construction ──

    Predicate operator&&(const Predicate& other) const {
        return logical_and(*this, other);
    }

    Predicate operator||(const Predicate& other) const {
        return logical_or(*this, other);
    }

    Predicate operator!() const {
        return logical_not(*this);
    }

    // ── Accessors ──

    PredicateNodeType type() const { return _type; }
    const std::vector<std::shared_ptr<Predicate>>& children() const { return _children; }

    gamax::expression::ExprTree& expr_tree() { return _expr; }
    const gamax::expression::ExprTree& expr_tree() const { return _expr; }
    select_options& expr_select_options() { return _select_opts; }
    const select_options& expr_select_options() const { return _select_opts; }

    gamax::size_type string_cid() const { return _string_cid; }
    gamax::string_impl_id string_impl() const { return _string_impl; }
    const gamax::string_op_options& string_options() const { return _string_opts; }

    const std::shared_ptr<gamax::string_scalar>& string_value() const { return _string_value; }
    const std::vector<std::shared_ptr<gamax::string_scalar>>& string_values() const { return _string_values; }

private:
    PredicateNodeType _type = PredicateNodeType::EXPR;
    std::vector<std::shared_ptr<Predicate>> _children;

    // Data for EXPR leaf
    gamax::expression::ExprTree _expr;
    select_options _select_opts;

    // Data for string predicate leaves
    gamax::size_type _string_cid = 0;
    gamax::string_impl_id _string_impl = gamax::ARROW_STRING;
    gamax::string_op_options _string_opts;
    std::shared_ptr<gamax::string_scalar> _string_value;
    std::vector<std::shared_ptr<gamax::string_scalar>> _string_values;
};
} // namespace gamax::select
