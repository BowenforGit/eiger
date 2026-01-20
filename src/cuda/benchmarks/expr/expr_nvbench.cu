#include <nvbench/nvbench.cuh>

#include <string>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>
#include <type_traits>
#include <sstream>

#include "expr/expr_fast.hpp"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "expr_builder.hpp"

struct expr_params {
    std::string expr_type;
    int64_t num_items;
    bool use_fuse;

    expr_params(nvbench::state& state) {
        expr_type = state.get_string("expr_type");
        num_items = state.get_int64("num_items");
        std::string impl = state.get_string("impl");
        use_fuse = (impl == "fuse");
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "expr_type: " << expr_type << ", "
            << "num_items: " << num_items << ", "
            << "use_fuse: " << (use_fuse ? "true" : "false");
        return oss.str();
    }
};

template<typename KeyT>
std::unique_ptr<ExpressionBuilder<KeyT>> get_expression_builder(
    const std::string& expr_type,
    gamax::size_type num_items)
{
    // Use fixed thresholds for consistent expression complexity testing
    // For uniform distribution from 0 to num_items-1
    KeyT mid = static_cast<KeyT>(num_items / 2);
    
    if (expr_type == "one_col_mult") {
        // 3 * col0
        return std::make_unique<SimpleMultiply<KeyT>>(static_cast<KeyT>(3));
    }
    else if (expr_type == "two_col_arith") {
        // col0 * (1 - col1)
        return std::make_unique<TwoColumnMultiply<KeyT>>();
    }
    else if (expr_type == "three_col_arith") {
        // col0 * (1 - col1) * (1 + col2)
        return std::make_unique<ThreeColumnMultiply<KeyT>>();
    }
    else if (expr_type == "four_col_arith") {
        // col0 * (1 - col1) - col2 * col3
        return std::make_unique<FourColumnArithmetic<KeyT>>();
    }
    else if (expr_type == "simple_gt") {
        // col0 > 134217728
        KeyT threshold = static_cast<KeyT>(134217728);
        return std::make_unique<SimpleGreaterThan<KeyT>>(threshold);
    }
    else if (expr_type == "two_col_and") {
        // col0 > t1 AND col1 < t2
        KeyT t1 = static_cast<KeyT>(161061273);
        KeyT t2 = static_cast<KeyT>(375809638);
        return std::make_unique<TwoColumnAndExpr<KeyT>>(t1, t2);
    }
    else if (expr_type == "two_col_and_or") {
        // (col0 == 0 AND col1 == 1) OR (col0 == 1 AND col1 == 0)
        return std::make_unique<TwoColumnAndOr<KeyT>>();
    }
    
    throw std::runtime_error("Unknown expression type: " + expr_type);
}

template<typename KeyT>
bool check_skip(expr_params const& params) {
    // No skipping needed for expression evaluation
    return false;
}

// =============================================================================
// Benchmark Function
// =============================================================================

template<typename KeyT>
void bench_expr(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem(); // ensure clean state

    expr_params params(state);

    if (check_skip<KeyT>(params)) {
        state.skip("Skipped configuration");
        return;
    }

    gamax::size_type num_items = params.num_items;
    
    // Get the expression builder
    auto expr_builder = get_expression_builder<KeyT>(params.expr_type, num_items);
    int ncols = expr_builder->num_columns();
    
    // Generate table with uniform distribution
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto table = gamax::datagen::device::pk_uniform_id(key_type, num_items, ncols, 42);
    
    // Build expression tree using the new API
    auto expr_tree = (*expr_builder)(table->view());
    
    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.add_element_count(num_items, "NumElements");
    // state.add_global_memory_reads<KeyT>(num_items * ncols, "DataSize");
    // state.collect_cupti_metrics();
    
    // Add expression info to summary
    nvbench::summary &summary = state.add_summary("expr_info");
    summary.set_string("name", "expr_info");
    summary.set_string("description", "The expression being evaluated");
    summary.set_string("value", expr_builder->to_string());

    auto fn = params.use_fuse ? 
        gamax::expression::evaluate_expr : 
        gamax::expression::evaluate_expr_nofuse;

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) { 
                auto res = fn(table->view(), expr_tree, launch.get_stream()); 
                cudaStreamSynchronize(launch.get_stream());
            });
    }
    catch(const std::exception& e) {
        std::cerr << "Error in expr benchmark: " << e.what() << '\n';
        std::ofstream out("expr_error.txt", std::ios::app);
        out << "Expression: " << expr_builder->to_string() << "\n";
        out << "Params: " << params.to_string() << "\n";
        out << "Error: " << e.what() << "\n\n";
    }
}

NVBENCH_BENCH_TYPES(bench_expr,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("expr")
.set_type_axes_names({"KeyType"})
.add_string_axis("impl", {"fuse", "nofuse"})
.add_string_axis("expr_type", {
    "one_col_mult",
    // "one_col_exp",
    "two_col_arith",
    "three_col_arith",
    "four_col_arith",
    "simple_gt",
    "two_col_and",
    "two_col_and_or",
})
.add_int64_power_of_two_axis("num_items", nvbench::range(20, 29, 1));
