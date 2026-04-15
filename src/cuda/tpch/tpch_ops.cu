#include <memory>
#include <cuda.h>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <nvtx3/nvtx3.hpp>

#include "tpch/tpch.hpp"
#include "memory/table.hpp"
#include "memory/column.hpp"
#include "types/data_type.hpp"
#include "operators/select/select.hpp"
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
    std::unique_ptr<gamax::table> tpch_select(
        query_context& qc,
        tpch_params const& params,
        std::string const& op_name,
        gamax::table_view input,
        gamax::expression::ExprTree& expr)
    {
        query_context::scoped_op op(qc, op_name);
        bool has_string_cols = false;
        for (gamax::size_type i = 0; i < input.num_columns(); i++) {
            if (input.get_column(i).type().id() == gamax::type_id::STRING) {
                has_string_cols = true;
                break;
            }
        }

        if(!has_string_cols) {
            return gamax::select::select(input, expr, params.select_options(op_name), params.stream);
        }

        // We need to materialize the strings
        auto mask = select_to_bitmask(input, expr, params.select_options(op_name), params.stream);
        std::vector<gamax::column_ptr> materialized_cols;
        gamax::column_ptr compact_tids {nullptr};
        for (gamax::size_type i = 0; i < input.num_columns(); i++) {
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
        std::string const& op_name,
        gamax::table_view key_view,
        gamax::table_view val_view,
        std::vector<gamax::groupby::aggregation_request>& agg_requests,
        gamax::groupby::groupby_hint& hint,
        cudaStream_t stream)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::groupby::groupby(key_view, val_view, agg_requests, hint, stream);
    }

    std::unique_ptr<gamax::table> tpch_aggregate(
        query_context& qc,
        std::string const& op_name,
        gamax::table_view input,
        std::vector<gamax::groupby::aggregation_request>& agg_requests,
        cudaStream_t stream)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::groupby::aggregate::aggregate(input, agg_requests, stream);
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
        std::string const& op_name,
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::join::inner_eq_join(left, right, hint, stream);
    }

    std::unique_ptr<gamax::table> tpch_semi_join(
        query_context& qc,
        std::string const& op_name,
        gamax::table_view left,
        gamax::table_view right,
        gamax::join::join_hint const& hint,
        cudaStream_t stream)
    {
        query_context::scoped_op op(qc, op_name);
        return gamax::join::semi_join(left, right, hint, stream);
    }
} // namespace gamax::tpch