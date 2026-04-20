#include <memory>
#include <cuda.h>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <numeric>
#include <nvtx3/nvtx3.hpp>

#include "tpch/tpch.hpp"
#include "tpch/tpch_ops.hpp"
#include "memory/table.hpp"
#include "memory/column.hpp"
#include "types/data_type.hpp"
#include "operators/select/select.hpp"
#include "operators/select/predicate.hpp"
#include "operators/project/project.hpp"
#include "operators/join/join.hpp"
#include "operators/join/inner_join.hpp"
#include "operators/join/semi_join.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/sort/sort.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "expr/expr_parser.hpp"
#include "utils/transform.hpp"

namespace gamax::tpch {
namespace {
    bool has_string_cols_from(gamax::table_view input, int from = 0) {
        for (int i = from; i < input.num_columns(); ++i) {
            if (input.get_column(i).type().id() == gamax::type_id::STRING) {
                return true;
            }
        }
        return false;
    }
} // namespace
    std::unique_ptr<gamax::table> tpch_select(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view input,
        gamax::select::Predicate& pred,
        std::vector<int> const& output_cols)
    {
        query_context::scoped_op op(qc, op_name);

        auto mask = select_to_bitmask(input, pred, params.stream);
        std::vector<gamax::column_ptr> materialized_cols;
        gamax::column_ptr compact_tids {nullptr};
        std::vector<int> actual_output_cols = output_cols;
        if(output_cols.empty()) {
            actual_output_cols.resize(input.num_columns());
            std::iota(actual_output_cols.begin(), actual_output_cols.end(), 0);
        }
        for (const auto& i : actual_output_cols) {
            if (input.get_column(i).type().id() == gamax::type_id::STRING) {
                if(!compact_tids) {
                    auto seqs = gamax::make_sequence<gamax::size_type>(input.num_rows(), 0, 1, params.stream);
                    compact_tids = gamax::util::device::compact(seqs->view(), mask->view(), params.stream);
                }
                gamax::string_column_view str_view(input.get_column(i), gamax::ARROW_STRING, params.stream);
                materialized_cols.push_back(gamax::gather_strings(str_view, compact_tids->view(), params.stream));
        } else {
                materialized_cols.push_back(gamax::util::device::compact(input.get_column(i), mask->view(), params.stream));
            }
        }

        return std::make_unique<gamax::table>(std::move(materialized_cols));
    }

    std::unique_ptr<gamax::table> tpch_select(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view input,
        gamax::expression::ExprTree& expr,
        std::vector<int> const& output_cols)
    {
        auto pred = gamax::select::Predicate::expr(expr, params.select_options(op_name));
        return tpch_select(qc, params, op_name, input, pred, output_cols);
    }

    std::unique_ptr<gamax::table> tpch_project(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view input,
        std::vector<gamax::expression::ExprTree>& exprs)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::project::project(input, exprs, params.project_options(op_name), params.stream);
    }

    std::unique_ptr<gamax::table> tpch_project(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        std::unique_ptr<gamax::table> input,
        std::vector<gamax::expression::ExprTree>& exprs)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::project::project(std::move(input), exprs, params.project_options(op_name), params.stream);
    }

    std::unique_ptr<gamax::table> tpch_groupby(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view key_view,
        gamax::table_view val_view,
        std::vector<gamax::groupby::aggregation_request>& agg_requests,
        gamax::groupby::groupby_hint& hint)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::groupby::groupby(key_view, val_view, agg_requests, hint, params.stream);
    }

    std::unique_ptr<gamax::table> tpch_aggregate(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view input,
        std::vector<gamax::groupby::aggregation_request>& agg_requests)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::groupby::aggregate::aggregate(input, agg_requests, params.stream);
    }

    std::unique_ptr<gamax::table> tpch_sort(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view input,
        std::vector<int> const& keys,
        std::vector<gamax::sort_order> const& orders)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::sort::sort_table(input, keys, orders, params.sort_options(op_name), params.stream);
    }

    std::unique_ptr<gamax::table> tpch_inner_join(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& hint)
    {
        query_context::scoped_op op(qc, op_name);
        if(!has_string_cols_from(left) && !has_string_cols_from(right)) {
            return gamax::join::inner_eq_join(left, right, hint, params.stream);
        }
        
        return gamax::join::inner_eq_join(left, right, hint, params.stream);
    }

    std::unique_ptr<gamax::table> tpch_semi_join(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& hint)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::join::semi_join(left, right, hint, params.stream);
    }
} // namespace gamax::tpch