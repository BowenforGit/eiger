#include "tpch/tpch.hpp"
#include "tpch/tpch_ops.hpp"
#include "expr/expr_parser.hpp"
#include "types/data_type.hpp"
#include "memory/table.hpp"
#include "operators/select/select.hpp"
#include "operators/project/project.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"

namespace gamax::tpch {
namespace {
    using namespace gamax::expression;
    using table_ptr = std::unique_ptr<gamax::table>;

    class TpchQ1Runner : public TpchQueryRunner {
    public:
        TpchQ1Runner(const tpch_params& params) : TpchQueryRunner(params) {
            reset();
            auto schema = get_schema(lineitem_->view());
            filter_expr_ = parse_expression_to_tree(
                "col0 <= 19980902",
                schema);

            project_exprs_ =
                ProjectBuilder(schema)
                .add("col1") // l_returnflag -> col0
                .add("col2") // l_linestatus -> col1
                .add("col3") // l_quantity -> col0
                .add("col4") // l_extendedprice -> col1
                .add("col5") // l_discount -> col2
                .add("col4 * (1 - col5)") // l_extendedprice * (1 - l_discount) -> col3
                .add("col4 * (1 - col5) * (1 + col6)") // l_extendedprice * (1 - l_discount) * (1 + l_tax) -> col4
                .get_exprs();

            agg_requests_ =
                AggregateBuilder()
                .add(0, gamax::groupby::aggregation_id::SUM) // sum(l_quantity)
                .add(1, gamax::groupby::aggregation_id::SUM) // sum(l_extendedprice)
                .add(3, gamax::groupby::aggregation_id::SUM) // sum(l_extendedprice * (1 - l_discount))
                .add(4, gamax::groupby::aggregation_id::SUM) // sum(l_extendedprice * (1 - l_discount) * (1 + l_tax))
                .add(0, gamax::groupby::aggregation_id::AVG) // avg(l_quantity)
                .add(1, gamax::groupby::aggregation_id::AVG) // avg(l_extendedprice)
                .add(2, gamax::groupby::aggregation_id::AVG) // avg(l_discount)
                .get_requests();
        }

        void reset() override {
            lineitem_ = reader_.load(
                "lineitem",
                {"l_shipdate", "l_returnflag", "l_linestatus", "l_quantity",
                    "l_extendedprice", "l_discount", "l_tax"});
        }

        std::unique_ptr<gamax::table> run() override {
            query_context qc("TPC-H Q1");

            PRINT_TABLE_PTR(lineitem_, 5, 0);

            auto filtered = tpch_select(qc, params_, "filter", lineitem_->view(), filter_expr_);
            
            PRINT_TABLE_PTR(filtered, 5, 0);
            
            auto projected = tpch_project(qc, params_, "project", std::move(filtered), project_exprs_);
            
            PRINT_TABLE_PTR(projected, 5, 0);
            
            // l_returnflag : R, N, A
            // l_linestatus : F, O
            std::vector<gamax::column_ptr> group_keys_cols;
            group_keys_cols.push_back(projected->remove_column(0)); // col0: l_returnflag
            group_keys_cols.push_back(projected->remove_column(0)); // col1: l_linestatus
            auto group_keys = gamax::make_table(std::move(group_keys_cols));
            auto groupby_hint = params_.groupby_options("groupby", true);
            auto aggregated = tpch_groupby(
                qc, params_, "groupby",
                group_keys->view(),
                projected->view(),
                agg_requests_,
                groupby_hint);
            
            PRINT_TABLE_PTR(aggregated, 5, 0);

            auto sorted = tpch_sort(
                qc, params_, "sort",
                aggregated->view(),
                {0, 1}, // l_returnflag, l_linestatus
                {gamax::sort_order::ASCENDING, gamax::sort_order::ASCENDING});
            
            PRINT_TABLE_PTR(sorted, 5, 0);

            return sorted;
        }

    private:
        table_ptr lineitem_{nullptr};
        ExprTree filter_expr_;
        std::vector<ExprTree> project_exprs_;
        std::vector<gamax::groupby::aggregation_request> agg_requests_;
    };

    class TpchQ6Runner : public TpchQueryRunner {
    public:
        TpchQ6Runner(const tpch_params& params) : TpchQueryRunner(params) {
            reset();
            auto schema = get_schema(lineitem_->view());
            // col0 >= 19940101 && col0 < 19950101 && col2 >= 0.05 && col2 <= 0.07 && col3 < 24
            filter_expr_ = parse_expression_to_tree(
                "col0 >= 8766 && col0 < 9131 && col2 >= 0.05 && col2 <= 0.07 && col3 < 24",
                schema);
            project_exprs_.push_back(parse_expression_to_tree("col1 * col2", schema));
            agg_requests_.push_back({0, gamax::groupby::aggregation_id::SUM});
        }

        void reset() override {
            lineitem_ = reader_.load(
                "lineitem",
                {"l_shipdate", "l_extendedprice", "l_discount", "l_quantity"});
        }

        std::unique_ptr<gamax::table> run() override {
            query_context qc("TPC-H Q6");

            PRINT_TABLE_PTR(lineitem_, 5, 0);

            auto filtered = tpch_select(qc, params_, "filter", lineitem_->view(), filter_expr_);

            PRINT_TABLE_PTR(filtered, 5, 0);

            auto projected = tpch_project(qc, params_, "project", filtered->view(), project_exprs_);
                
            PRINT_TABLE_PTR(projected, 5, 0);
            
            auto aggregate = tpch_aggregate(qc, params_, "aggregate", projected->view(), agg_requests_);

            PRINT_TABLE_PTR(aggregate, 5, 0);

            return aggregate;
        }

    private:
        table_ptr lineitem_{nullptr};
        ExprTree filter_expr_;
        std::vector<ExprTree> project_exprs_;
        std::vector<gamax::groupby::aggregation_request> agg_requests_;
    };

    class TpchQ14Runner : public TpchQueryRunner {
    public:
        TpchQ14Runner(const tpch_params& params) : TpchQueryRunner(params) {
            reset();
            auto lineitem_schema = get_schema(lineitem_->view());
            // col0 >= 1995-09-01 && col0 < 1995-10-01
            filter_lineitem_expr_ = parse_expression_to_tree(
                "col0 >= 9374 && col0 < 9404",
                lineitem_schema);
            project_lineitem_exprs_ =
                ProjectBuilder(lineitem_schema)
                .add("col1") // l_partkey -> col0
                .add("col2 * (1 - col3)") // l_extendedprice * (1 - l_discount) -> col1
                .get_exprs();
            agg_requests_ = 
                AggregateBuilder()
                .add(0, gamax::groupby::aggregation_id::SUM) // sum(promo_revenue)
                .add(1, gamax::groupby::aggregation_id::SUM) // sum(revenue)
                .get_requests();
        }

        void reset() override {
            lineitem_ = reader_.load(
                "lineitem",
                {"l_shipdate" /*col0*/, "l_partkey" /*col1*/, "l_extendedprice" /*col2*/, "l_discount" /*col3*/});
            part_ = reader_.load(
                "part",
                {"p_partkey" /*col0*/, "p_type" /*col1*/});
        }

        std::unique_ptr<gamax::table> run() override {
            query_context qc("TPC-H Q14");

            PRINT_TABLE_PTR(lineitem_, 5, 0);

            auto filtered_lineitem = tpch_select(qc, params_, "filter_lineitem", lineitem_->view(), filter_lineitem_expr_);

            PRINT_TABLE_PTR(filtered_lineitem, 5, 0);

            auto projected_lineitem = tpch_project(qc, params_, "project_lineitem", filtered_lineitem->view(), project_lineitem_exprs_);
                
            PRINT_TABLE_PTR(projected_lineitem, 5, 0);

            auto join_option = JoinOption(params_, "join_lineitem_part").unique_right().get();

            // partkey, revenue, p_type
            // TODO: currently join cannot materialize the string columns
            auto join_lineitem_part = tpch_inner_join(qc, params_, "join_lineitem_part", projected_lineitem->view(), part_->view(), join_option);

            PRINT_TABLE_PTR(join_lineitem_part, 5, 0);
            
            auto projected_join_lineitem_part = project_join_lineitem_part(qc, std::move(join_lineitem_part));

            PRINT_TABLE_PTR(projected_join_lineitem_part, 5, 0);
            
            auto aggregate = tpch_aggregate(qc, params_, "aggregate", projected_join_lineitem_part->view(), agg_requests_);

            PRINT_TABLE_PTR(aggregate, 5, 0);

            return aggregate;
        }
    
    private:
        table_ptr project_join_lineitem_part(query_context& qc, table_ptr input) {
            query_context::scoped_op op(qc, "project_join_lineitem_part");
            auto scalar = gamax::make_string_scalar("PROMO", params_.stream);
            auto pred = gamax::select::Predicate::string_startswith(2 /*col2 - p_type */, std::move(scalar));
            auto mask = gamax::select::select_to_bitmask(input->view(), pred, params_.stream);
            input->add_column(std::move(mask)); // col3: startswith(p_type, 'PROMO')

            if(project_join_lineitem_part_exprs_.empty()) {
                auto schema = get_schema(input->view());
                project_join_lineitem_part_exprs_ =
                    ProjectBuilder(schema)
                    .add("col3 ? col1 : 0.0")
                    .add("col1") // l_extendedprice * (1 - l_discount) -> col1
                    .get_exprs();
            }

            return tpch_project(qc, params_, "project_lineitem", std::move(input), project_join_lineitem_part_exprs_);
        }

    private:
        table_ptr lineitem_{nullptr};
        table_ptr part_{nullptr};
        ExprTree filter_lineitem_expr_;
        std::vector<ExprTree> project_lineitem_exprs_;
        std::vector<ExprTree> project_join_lineitem_part_exprs_;
        std::vector<gamax::groupby::aggregation_request> agg_requests_;
    };
}

std::unique_ptr<TpchQueryRunner> get_query_runner(tpch_query query_id, const tpch_params& params) {
    switch(query_id) {
        case tpch_query::Q1: return std::make_unique<TpchQ1Runner>(params);
        case tpch_query::Q6: return std::make_unique<TpchQ6Runner>(params);
        case tpch_query::Q14: return std::make_unique<TpchQ14Runner>(params);
        default:
            throw std::runtime_error("Unsupported TPC-H query id: " + std::to_string(static_cast<int>(query_id)));
    }
}
} // namespace gamax::tpch
