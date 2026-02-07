#include <nvbench/nvbench.cuh>

#include <string>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>
#include <type_traits>

#include "operators/select/select.hpp"
#include "expr/expr_fast.hpp"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "../expr/expr_builder.hpp"

struct select_params {
    std::string expr_type;
    int64_t num_items;
    double selectivity;
    bool use_fuse;

    select_params(nvbench::state& state) {
        expr_type = state.get_string("expr_type");
        num_items = state.get_int64("num_items");
        selectivity = state.get_float64("selectivity");
        std::string impl = state.get_string("impl");
        use_fuse = (impl == "fuse");
    }
};

template<typename KeyT>
std::unique_ptr<ExpressionBuilder<KeyT>> get_expression_builder(
    const std::string& expr_type,
    double selectivity,
    gamax::size_type num_items)
{
    // Calculate thresholds based on selectivity
    // For uniform distribution from 0 to num_items-1
    KeyT mid = static_cast<KeyT>(num_items / 2);
    KeyT range = static_cast<KeyT>(num_items * selectivity / 2);
    
    if (expr_type == "simple_gt") {
        // col0 > threshold (selects approximately selectivity fraction)
        KeyT threshold = static_cast<KeyT>(num_items * (1.0 - selectivity));
        return std::make_unique<SimpleGreaterThan<KeyT>>(threshold);
    }
    else if (expr_type == "simple_ge") {
        KeyT threshold = static_cast<KeyT>(num_items * (1.0 - selectivity));
        return std::make_unique<SimpleGreaterEqual<KeyT>>(threshold);
    }
    else if (expr_type == "simple_lt") {
        KeyT threshold = static_cast<KeyT>(num_items * selectivity);
        return std::make_unique<SimpleLessThan<KeyT>>(threshold);
    }
    else if (expr_type == "range_and") {
        // col0 > low AND col0 < high
        KeyT low = mid - range;
        KeyT high = mid + range;
        return std::make_unique<RangeAndExpr<KeyT>>(low, high);
    }
    else if (expr_type == "range_or") {
        // col0 < low OR col0 > high
        KeyT low = static_cast<KeyT>(num_items * selectivity / 2);
        KeyT high = static_cast<KeyT>(num_items * (1.0 - selectivity / 2));
        return std::make_unique<RangeOrExpr<KeyT>>(low, high);
    }
    else if (expr_type == "two_col_sum") {
        // col0 + col1 > threshold
        KeyT threshold = static_cast<KeyT>(num_items * 2 * (1.0 - selectivity));
        return std::make_unique<TwoColumnSum<KeyT>>(threshold);
    }
    else if (expr_type == "two_col_and") {
        // col0 > t1 AND col1 < t2
        KeyT t1 = static_cast<KeyT>(num_items * 0.3);
        KeyT t2 = static_cast<KeyT>(num_items * 0.7);
        return std::make_unique<TwoColumnAndExpr<KeyT>>(t1, t2);
    }
    else if (expr_type == "two_col_or") {
        // col0 <= t1 OR col1 >= t2
        KeyT t1 = static_cast<KeyT>(num_items * selectivity / 2);
        KeyT t2 = static_cast<KeyT>(num_items * (1.0 - selectivity / 2));
        return std::make_unique<TwoColumnOrExpr<KeyT>>(t1, t2);
    }
    else if (expr_type == "three_col_complex") {
        // (col0 + col1) * col2 > threshold
        KeyT threshold = static_cast<KeyT>(num_items * num_items * 2 * (1.0 - selectivity));
        return std::make_unique<ThreeColumnComplexExpr<KeyT>>(threshold);
    }
    else if (expr_type == "three_col_and") {
        // col0 > t1 AND col1 < t2 AND col2 >= t3
        KeyT t1 = static_cast<KeyT>(num_items * 0.2);
        KeyT t2 = static_cast<KeyT>(num_items * 0.8);
        KeyT t3 = static_cast<KeyT>(num_items * 0.4);
        return std::make_unique<ThreeColumnAndExpr<KeyT>>(t1, t2, t3);
    }
    else if (expr_type == "four_col_equal") {
        return std::make_unique<FourColumnEqualExpr<KeyT>>();
    }
    else if (expr_type == "two_col_sum_op") {
        // Using operator overloading
        KeyT threshold = static_cast<KeyT>(num_items * 2 * (1.0 - selectivity));
        return std::make_unique<TwoColumnSumOperator<KeyT>>(threshold);
    }
    else if (expr_type == "range_and_op") {
        // Using operator overloading
        KeyT low = mid - range;
        KeyT high = mid + range;
        return std::make_unique<RangeAndOperator<KeyT>>(low, high);
    }
    else if (expr_type == "three_col_complex_op") {
        // Using operator overloading
        KeyT threshold = static_cast<KeyT>(num_items * num_items * 2 * (1.0 - selectivity));
        return std::make_unique<ThreeColumnComplexOperator<KeyT>>(threshold);
    }
    else if (expr_type == "shift_op") {
        KeyT threshold = static_cast<KeyT>(num_items / 4 * (1.0 - selectivity));
        return std::make_unique<ShiftOperator<KeyT>>(threshold);
    }
    
    throw std::runtime_error("Unknown expression type: " + expr_type);
}

template<typename KeyT>
bool check_skip(select_params const& params) {
    // Skip certain combinations to reduce benchmark matrix size
    // For example, skip non-primary configurations
    
    // Only test selectivity variations for simple expressions
    if (params.expr_type.find("simple") == std::string::npos && 
        params.expr_type.find("range") == std::string::npos) {
        if (params.selectivity != 0.5) {
            return true;
        }
    }
    
    return false;
}

// =============================================================================
// Benchmark Function
// =============================================================================

template<typename KeyT>
void bench_select_fast(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem(); // ensure clean state

    select_params params(state);

    if (check_skip<KeyT>(params)) {
        state.skip("Skipped configuration");
        return;
    }

    gamax::size_type num_items = params.num_items;
    
    // Get the expression builder
    auto expr_builder = get_expression_builder<KeyT>(params.expr_type, params.selectivity, num_items);
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

    auto fn = gamax::select::select;
    gamax::select::select_options options;
    options.fuse_operations = params.use_fuse;

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) { 
                auto res = fn(table->view(), expr_tree, options, launch.get_stream()); 
                cudaStreamSynchronize(launch.get_stream());
            });
    }
    catch(const std::exception& e) {
        std::cerr << "Error in select_fast benchmark: " << e.what() << '\n';
        std::ofstream out("select_fast_error.txt", std::ios::app);
        out << "Expression: " << expr_builder->to_string() << "\n";
        out << "Error: " << e.what() << "\n\n";
    }
}

NVBENCH_BENCH_TYPES(bench_select_fast,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("select_fast")
.set_type_axes_names({"KeyType"})
.add_string_axis("impl", {"fuse", "nofuse"})
.add_string_axis("expr_type", {
    "simple_gt",
    "simple_ge", 
    "simple_lt",
    "range_and",
    "range_or",
    "two_col_sum",
    "two_col_and",
    "two_col_or",
    "three_col_complex",
    "three_col_and",
    "four_col_equal",
    "two_col_sum_op",
    "range_and_op",
    "three_col_complex_op",
    "shift_op"
})
.add_int64_power_of_two_axis("num_items", nvbench::range(20, 27, 1))
.add_float64_axis("selectivity", {0.01, 0.1, 0.25, 0.5, 0.75, 0.9});
