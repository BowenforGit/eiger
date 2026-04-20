#pragma once

#include <memory>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <stdexcept>

#include "tpch/tpch.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "expr/expr_fast.hpp"
#include "operators/select/predicate.hpp"

#define TPCH_DEBUG

namespace gamax::tpch {

// ---------------------------------------------------------------------------
// Operator wrappers: handle NVTX push/pop and algorithm lookup
// ---------------------------------------------------------------------------
std::unique_ptr<gamax::table> tpch_select(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view input,
    gamax::expression::ExprTree& expr,
    std::vector<int> const& output_cols = {});

std::unique_ptr<gamax::table> tpch_select(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view input,
    gamax::select::Predicate& pred,
    std::vector<int> const& output_cols = {});

std::unique_ptr<gamax::table> tpch_project(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view input,
    std::vector<gamax::expression::ExprTree>& exprs);

std::unique_ptr<gamax::table> tpch_project(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    std::unique_ptr<gamax::table> input,
    std::vector<gamax::expression::ExprTree>& exprs);

std::unique_ptr<gamax::table> tpch_groupby(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view key_view,
    gamax::table_view val_view,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint);

std::unique_ptr<gamax::table> tpch_aggregate(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view input,
    std::vector<gamax::groupby::aggregation_request>& agg_requests);

std::unique_ptr<gamax::table> tpch_sort(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view input,
    std::vector<int> const& keys,
    std::vector<gamax::sort_order> const& orders);

std::unique_ptr<gamax::table> tpch_inner_join(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint);

std::unique_ptr<gamax::table> tpch_semi_join(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint);
} // namespace gamax::tpch