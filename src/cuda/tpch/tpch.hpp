#pragma once

#include <memory>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <stack>
#include <string>
#include <stdexcept>
#include <nvtx3/nvtx3.hpp>

#include "memory/table.hpp"
#include "operators/select/select.hpp"
#include "operators/project/project.hpp"
#include "operators/join/join.hpp"
#include "operators/join/inner_join.hpp"
#include "operators/join/semi_join.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/sort/sort.hpp"
#include "expr/expr_parser.hpp"

#define TPCH_DEBUG

namespace gamax::tpch {
enum class tpch_query {
    Q1 = 1,
    Q2,
    Q3,
    Q4,
    Q5,
    Q6,
    Q7,
    Q8,
    Q9,
    Q10,
    Q11,
    Q12,
    Q13,
    Q14,
    Q15,
    Q16,
    Q17,
    Q18,
    Q19,
    Q20,
    Q21,
    Q22
};

struct query_context {
    explicit query_context(const std::string& query) {
        nvtxRangePushA(query.c_str());
    }

    ~query_context() {
        if (!op_stack_.empty()) {
           throw std::runtime_error("Operation stack is not empty on query_context destruction. Remaining operations: " + std::to_string(op_stack_.size()));
        }
        nvtxRangePop();
    }

    void push_op(const std::string& op) {
        op_stack_.push(op);
        nvtxRangePushA(op.c_str());
    }

    const std::string& cur_op() const {
        return op_stack_.top();
    }

    void pop_op() {
        if (!op_stack_.empty()) {
            nvtxRangePop();
            op_stack_.pop();
        } else {
            throw std::runtime_error("Operation stack is empty, cannot pop");
        }
    }

    // RAII guard: pushes on construction, pops on destruction
    struct scoped_op {
        query_context& qc;
        scoped_op(query_context& qc, const std::string& name) : qc(qc) { qc.push_op(name); }
        ~scoped_op() { qc.pop_op(); }
        scoped_op(const scoped_op&) = delete;
        scoped_op& operator=(const scoped_op&) = delete;
    };

private:
    std::stack<std::string> op_stack_;
};

struct tpch_params {
    gamax::size_type scale_factor; // in GB
    std::unordered_map<std::string, int> algo_map;
    cudaStream_t stream = cudaStreamDefault;

    tpch_params(gamax::size_type sf) : scale_factor(sf), algo_map() {}

    void add_algo(std::string op, int algo_id) {
        algo_map[std::move(op)] = algo_id;
    }

    int get_algo(const std::string& op) const {
        auto it = algo_map.find(op);
        if (it != algo_map.end()) {
            return it->second;
        }
        throw std::runtime_error("Algorithm for operation " + op + " not found");
    }

    gamax::select::select_options select_options(const std::string& op_name) const {
        gamax::select::select_options options;
        options.fuse_operations = static_cast<bool>(get_algo(op_name));
        return options;
    }

    gamax::project::project_options project_options(const std::string& op_name) const {
        gamax::project::project_options options;
        options.fuse_operations = static_cast<bool>(get_algo(op_name));
        return options;
    }

    gamax::join::join_hint join_options(const std::string& op_name) const {
        gamax::join::join_hint options;
        int composite = get_algo(op_name);
        options.join_algo = static_cast<gamax::join::join_algo>(composite / 100);
        options.mat_algo = static_cast<gamax::join::materialization_algo>(composite % 100);
        return options;
    }

    gamax::groupby::groupby_hint groupby_options(const std::string& op_name) const {
        gamax::groupby::groupby_hint options;
        options.groupby_algo = static_cast<gamax::groupby::groupby_algo>(get_algo(op_name));
        return options;
    }

    gamax::sort::sort_table_options sort_options(const std::string& op_name) const {
        gamax::sort::sort_table_options options;
        options.smart_keys_fusion = static_cast<bool>(get_algo(op_name));
        return options;
    }

    static int parse_join_composite(const std::string& combo) {
        static const std::unordered_map<std::string, int> join_combos = {
            {"hash_auto", 100}, {"hash_gfur", 101}, {"hash_gftr", 102},
            {"sort_gfur", 201}, {"sort_gftr", 202},
            {"partition_gfur", 301}, {"partition_gftr", 302},
        };
        auto it = join_combos.find(combo);
        if (it == join_combos.end()) {
            throw std::runtime_error("Unknown join combo: " + combo);
        }
        return it->second;
    }

    // Unified parser: maps any operator axis value string to an algo int.
    // Works for all operator types — the axis name selects the operator,
    // and this function converts the value string to the corresponding int.
    static int parse_algo_value(const std::string& value) {
        static const std::unordered_map<std::string, int> algo_values = {
            // fuse/nofuse (select, project)
            {"fuse", 1}, {"nofuse", 0},
            // sort
            {"skf", 1}, {"no_skf", 0},
            // groupby
            {"hash",            static_cast<int>(gamax::groupby::HASH)},
            {"sort_gftr",       static_cast<int>(gamax::groupby::SORT)},
            {"sort_gftr_opt",   static_cast<int>(gamax::groupby::SORT_OPT)},
            {"sort_gfur",       static_cast<int>(gamax::groupby::SORT_GFUR)},
            {"partition_gftr",  static_cast<int>(gamax::groupby::PARTITION)},
            {"partition_gfur",  static_cast<int>(gamax::groupby::PARTITION_GFUR)},
            // join (composite encoding: join_algo * 100 + mat_algo)
            {"hash_auto", 100}, {"hash_gfur", 101}, {"hash_gftr", 102},
            {"sort_merge_gfur", 201}, {"sort_merge_gftr", 202},
            {"partition_hash_gfur", 301}, {"partition_hash_gftr", 302},
        };
        auto it = algo_values.find(value);
        if (it == algo_values.end()) {
            throw std::runtime_error("Unknown operator algo value: " + value);
        }
        return it->second;
    }
};

class TpchTableReader {
public:
    explicit TpchTableReader(const tpch_params& params);
    ~TpchTableReader();

    // Non-copyable, movable
    TpchTableReader(const TpchTableReader&) = delete;
    TpchTableReader& operator=(const TpchTableReader&) = delete;
    TpchTableReader(TpchTableReader&&) noexcept;
    TpchTableReader& operator=(TpchTableReader&&) noexcept;

    std::unique_ptr<gamax::table> load(
        const std::string& table_name,
        std::vector<std::string> column_names);

private:
    tpch_params params_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Helpers for building query plans
// ---------------------------------------------------------------------------
inline std::vector<gamax::data_type> get_schema(gamax::table_view table) {
    std::vector<gamax::data_type> schema;
    for (int i = 0; i < table.num_columns(); ++i) {
        schema.push_back(table[i].type());
    }
    return schema;
}

class ProjectBuilder {
public:
    ProjectBuilder(std::vector<gamax::data_type> const& schema) : schema_(schema) {}

    ProjectBuilder& add(std::string const& expr_str) {
        exprs_.push_back(gamax::expression::parse_expression_to_tree(expr_str, schema_));
        return *this;
    }

    std::vector<gamax::expression::ExprTree> get_exprs() {
        return std::move(exprs_);
    }

private:
    std::vector<gamax::data_type> schema_;
    std::vector<gamax::expression::ExprTree> exprs_;
};

class AggregateBuilder {
public:
    AggregateBuilder() = default;

    AggregateBuilder& add(int column_index, gamax::groupby::aggregation_id agg_id) {
        agg_requests_.push_back({column_index, agg_id});
        return *this;
    }

    std::vector<gamax::groupby::aggregation_request> get_requests() {
        return std::move(agg_requests_);
    }

private:
    std::vector<gamax::groupby::aggregation_request> agg_requests_;
};

// ---------------------------------------------------------------------------
// Operator wrappers: handle NVTX push/pop and algorithm lookup
// ---------------------------------------------------------------------------
std::unique_ptr<gamax::table> tpch_select(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view input,
    gamax::expression::ExprTree& expr);

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
    std::string const& op_name,
    gamax::table_view key_view,
    gamax::table_view val_view,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream);

std::unique_ptr<gamax::table> tpch_aggregate(
    query_context& qc,
    std::string const& op_name,
    gamax::table_view input,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    cudaStream_t stream);

std::unique_ptr<gamax::table> tpch_sort(
    query_context& qc,
    tpch_params const& params,
    std::string const& op_name,
    gamax::table_view input,
    std::vector<int> const& keys,
    std::vector<gamax::sort_order> const& orders);

std::unique_ptr<gamax::table> tpch_inner_join(
    query_context& qc,
    std::string const& op_name,
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream);

std::unique_ptr<gamax::table> tpch_semi_join(
    query_context& qc,
    std::string const& op_name,
    gamax::table_view left,
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Query runner base and factory
// ---------------------------------------------------------------------------
class TpchQueryRunner {
public:
    explicit TpchQueryRunner(const tpch_params& params) : params_(params), reader_(params) {}
    virtual ~TpchQueryRunner() = default;

    virtual void reset() = 0;
    virtual std::unique_ptr<gamax::table> run() = 0;

protected:
    tpch_params params_;
    TpchTableReader reader_;
};

std::unique_ptr<TpchQueryRunner> get_query_runner(tpch_query qid, const tpch_params& params);

void print_table(const gamax::table_view& view, gamax::size_type nrows, gamax::size_type offset);

#ifdef TPCH_DEBUG
#define PRINT_TABLE_PTR(table, nrows, offset) \
    do { \
        std::cout << "\n" << __FILE__ << ":" << __LINE__ << " - "; \
        std::cout << "Table: " << #table << std::endl; \
        print_table(table->view(), nrows, offset); \
    } while(0)
#else
#define PRINT_TABLE_PTR(table, nrows, offset) do {} while(0)
#endif // TPCH_DEBUG
} // namespace gamax::tpch