// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "predicate.hpp"

#include <stdexcept>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "utils/transform.hpp"

namespace gamax::select {
namespace {

gamax::column_ptr evaluate_predicate(
    gamax::table_view table,
    Predicate& pred,
    select_options options,
    cudaStream_t stream)
{
    switch (pred.type()) {
        case PredicateNodeType::EXPR: {
            return gamax::select::select_to_bitmask(
                table, pred.expr_tree(), stream, options);
        }

        case PredicateNodeType::STRING_EQUAL: {
            gamax::string_column_view str_col(
                table.get_column(pred.string_cid()), pred.string_impl());
            return gamax::equal(
                str_col, pred.string_value()->view(), stream, pred.string_options());
        }

        case PredicateNodeType::STRING_STARTSWITH: {
            gamax::string_column_view str_col(
                table.get_column(pred.string_cid()), pred.string_impl());
            return gamax::startswith(
                str_col, pred.string_value()->view(), stream, pred.string_options());
        }

        case PredicateNodeType::STRING_ENDSWITH: {
            gamax::string_column_view str_col(
                table.get_column(pred.string_cid()), pred.string_impl());
            return gamax::endswith(
                str_col, pred.string_value()->view(), stream, pred.string_options());
        }

        case PredicateNodeType::STRING_CONTAINS: {
            gamax::string_column_view str_col(
                table.get_column(pred.string_cid()), pred.string_impl());
            return gamax::contains(
                str_col, pred.string_value()->view(), stream, pred.string_options());
        }

        case PredicateNodeType::STRING_CONTAINS_MULTIPLE: {
            gamax::string_column_view str_col(
                table.get_column(pred.string_cid()), pred.string_impl());
            std::vector<gamax::string_scalar_view> views;
            views.reserve(pred.string_values().size());
            for (const auto& s : pred.string_values()) {
                views.push_back(s->view());
            }
            return gamax::contains_multiple(
                str_col, views, stream, pred.string_options());
        }

        case PredicateNodeType::STRING_IN: {
            gamax::string_column_view str_col(
                table.get_column(pred.string_cid()), pred.string_impl());
            std::vector<gamax::string_scalar_view> views;
            views.reserve(pred.string_values().size());
            for (const auto& s : pred.string_values()) {
                views.push_back(s->view());
            }
            return gamax::in_multiple(
                str_col, views, stream, pred.string_options());
        }

        case PredicateNodeType::AND: {
            auto lhs = evaluate_predicate(table, *pred.children()[0], options, stream);
            auto rhs = evaluate_predicate(table, *pred.children()[1], options, stream);
            return gamax::util::device::bitmask_and(lhs->view(), rhs->view(), stream);
        }

        case PredicateNodeType::OR: {
            auto lhs = evaluate_predicate(table, *pred.children()[0], options, stream);
            auto rhs = evaluate_predicate(table, *pred.children()[1], options, stream);
            return gamax::util::device::bitmask_or(lhs->view(), rhs->view(), stream);
        }

        case PredicateNodeType::NOT: {
            auto child = evaluate_predicate(table, *pred.children()[0], options, stream);
            return gamax::util::device::bitmask_not(child->view(), stream);
        }

        default:
            throw std::runtime_error("Unknown predicate node type");
    }
}

} // namespace

gamax::column_ptr select_to_bitmask(
    gamax::table_view table,
    Predicate& predicate,
    cudaStream_t stream,
    select_options options)
{
    return evaluate_predicate(table, predicate, options, stream);
}

} // namespace gamax::select
