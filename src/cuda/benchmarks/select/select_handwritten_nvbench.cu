#include <nvbench/nvbench.cuh>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include "../common/nvtx_stub.hpp"

#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>

#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"

// =============================================================================
// Handwritten Expression Evaluation Kernels
// =============================================================================

// Simple greater than: col0 > threshold
template<typename KeyT>
struct SimpleGTHandwritten {
    KeyT threshold;
    
    SimpleGTHandwritten(KeyT thresh) : threshold(thresh) {}
    
    std::string to_string() const {
        return "col0 > " + std::to_string(threshold) + " (handwritten)";
    }
    
    int num_columns() const { return 1; }
    
    // Evaluation function using thrust::transform
    void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const {
        gamax::size_type num_rows = table.num_rows();
        
        // Get column data pointer
        auto col0_data = table[0].data<KeyT>();
        auto output_data = output.data<bool>();
        
        // Capture threshold by value to avoid device accessing host memory
        KeyT threshold_val = threshold;
        
        // Use thrust::transform with counting iterator
        thrust::counting_iterator<gamax::size_type> first(0);
        thrust::counting_iterator<gamax::size_type> last = first + num_rows;
        
        nvtxRangePushA("SimpleGT_evaluate");
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            first, 
            last,
            output_data,
            [=] __device__ (gamax::size_type idx) {
                return col0_data[idx] > threshold_val;
            }
        );
        nvtxRangePop();
    }
};

// Range AND: col0 > low AND col0 < high
template<typename KeyT>
struct RangeANDHandwritten {
    KeyT low, high;
    
    RangeANDHandwritten(KeyT l, KeyT h) : low(l), high(h) {}
    
    std::string to_string() const {
        return "col0 > " + std::to_string(low) + " AND col0 < " + std::to_string(high) + " (handwritten)";
    }
    
    int num_columns() const { return 1; }
    
    // Evaluation function using thrust::transform
    void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const {
        gamax::size_type num_rows = table.num_rows();
        
        // Get column data pointer
        auto col0_data = table[0].data<KeyT>();
        auto output_data = output.data<bool>();
        
        // Capture low and high by value to avoid device accessing host memory
        KeyT low_val = low;
        KeyT high_val = high;
        
        // Use thrust::transform with counting iterator
        thrust::counting_iterator<gamax::size_type> first(0);
        thrust::counting_iterator<gamax::size_type> last = first + num_rows;
        
        nvtxRangePushA("RangeAND_evaluate");
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            first, 
            last,
            output_data,
            [=] __device__ (gamax::size_type idx) {
                KeyT val = col0_data[idx];
                return (val > low_val) && (val < high_val);
            }
        );
        nvtxRangePop();
    }
};

// Two column sum: col0 + col1 > threshold
template<typename KeyT>
struct TwoColSumHandwritten {
    KeyT threshold;
    
    TwoColSumHandwritten(KeyT thresh) : threshold(thresh) {}
    
    std::string to_string() const {
        return "col0 + col1 > " + std::to_string(threshold) + " (handwritten)";
    }
    
    int num_columns() const { return 2; }
    
    // Evaluation function using thrust::transform
    void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const {
        gamax::size_type num_rows = table.num_rows();
        
        // Get column data pointers
        auto col0_data = table[0].data<KeyT>();
        auto col1_data = table[1].data<KeyT>();
        auto output_data = output.data<bool>();
        
        // Capture threshold by value to avoid device accessing host memory
        KeyT threshold_val = threshold;
        
        // Use thrust::transform with counting iterator
        thrust::counting_iterator<gamax::size_type> first(0);
        thrust::counting_iterator<gamax::size_type> last = first + num_rows;
        
        nvtxRangePushA("TwoColSum_evaluate");
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            first, 
            last,
            output_data,
            [=] __device__ (gamax::size_type idx) {
                return (col0_data[idx] + col1_data[idx]) > threshold_val;
            }
        );
        nvtxRangePop();
    }
};

// Four columns: (col0 + col1) == (col2 + col3)
template<typename KeyT>
struct FourColEqualHandwritten {
    FourColEqualHandwritten() {}
    
    std::string to_string() const {
        return "(col0 + col1) == (col2 + col3) (handwritten)";
    }
    
    int num_columns() const { return 4; }
    
    // Evaluation function using thrust::transform
    void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const {
        gamax::size_type num_rows = table.num_rows();
        
        // Get column data pointers
        auto col0_data = table[0].data<KeyT>();
        auto col1_data = table[1].data<KeyT>();
        auto col2_data = table[2].data<KeyT>();
        auto col3_data = table[3].data<KeyT>();
        auto output_data = output.data<bool>();
        
        // Use thrust::transform with counting iterator
        thrust::counting_iterator<gamax::size_type> first(0);
        thrust::counting_iterator<gamax::size_type> last = first + num_rows;
        
        nvtxRangePushA("FourColEqual_evaluate");
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            first, 
            last,
            output_data,
            [=] __device__ (gamax::size_type idx) {
                return (col0_data[idx] + col1_data[idx]) == (col2_data[idx] + col3_data[idx]);
            }
        );
        nvtxRangePop();
    }
};

// =============================================================================
// Benchmark Parameters and Utilities
// =============================================================================

struct select_handwritten_params {
    std::string expr_type;
    int64_t num_items;
    double selectivity;

    select_handwritten_params(nvbench::state& state) {
        expr_type = state.get_string("expr_type");
        num_items = state.get_int64("num_items");
        selectivity = state.get_float64("selectivity");
    }
};

// Base class for handwritten expression evaluators
template<typename KeyT>
struct HandwrittenExprEvaluator {
    virtual ~HandwrittenExprEvaluator() = default;
    virtual void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const = 0;
    virtual std::string to_string() const = 0;
    virtual int num_columns() const = 0;
};

// Wrapper for SimpleGTHandwritten
template<typename KeyT>
struct SimpleGTEvaluator : public HandwrittenExprEvaluator<KeyT> {
    SimpleGTHandwritten<KeyT> impl;
    
    SimpleGTEvaluator(KeyT threshold) : impl(threshold) {}
    
    void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const override {
        impl.evaluate(table, output, stream);
    }
    
    std::string to_string() const override {
        return impl.to_string();
    }
    
    int num_columns() const override {
        return impl.num_columns();
    }
};

// Wrapper for RangeANDHandwritten
template<typename KeyT>
struct RangeANDEvaluator : public HandwrittenExprEvaluator<KeyT> {
    RangeANDHandwritten<KeyT> impl;
    
    RangeANDEvaluator(KeyT low, KeyT high) : impl(low, high) {}
    
    void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const override {
        impl.evaluate(table, output, stream);
    }
    
    std::string to_string() const override {
        return impl.to_string();
    }
    
    int num_columns() const override {
        return impl.num_columns();
    }
};

// Wrapper for TwoColSumHandwritten
template<typename KeyT>
struct TwoColSumEvaluator : public HandwrittenExprEvaluator<KeyT> {
    TwoColSumHandwritten<KeyT> impl;
    
    TwoColSumEvaluator(KeyT threshold) : impl(threshold) {}
    
    void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const override {
        impl.evaluate(table, output, stream);
    }
    
    std::string to_string() const override {
        return impl.to_string();
    }
    
    int num_columns() const override {
        return impl.num_columns();
    }
};

// Wrapper for FourColEqualHandwritten
template<typename KeyT>
struct FourColEqualEvaluator : public HandwrittenExprEvaluator<KeyT> {
    FourColEqualHandwritten<KeyT> impl;
    
    FourColEqualEvaluator() {}
    
    void evaluate(gamax::table_view table, gamax::mutable_column_view output, cudaStream_t stream) const override {
        impl.evaluate(table, output, stream);
    }
    
    std::string to_string() const override {
        return impl.to_string();
    }
    
    int num_columns() const override {
        return impl.num_columns();
    }
};

template<typename KeyT>
std::unique_ptr<HandwrittenExprEvaluator<KeyT>> get_handwritten_evaluator(
    const std::string& expr_type,
    double selectivity,
    gamax::size_type num_items)
{
    // Calculate thresholds based on selectivity
    KeyT mid = static_cast<KeyT>(num_items / 2);
    KeyT range = static_cast<KeyT>(num_items * selectivity / 2);
    
    if (expr_type == "simple_gt") {
        // col0 > threshold (selects approximately selectivity fraction)
        KeyT threshold = static_cast<KeyT>(num_items * (1.0 - selectivity));
        return std::make_unique<SimpleGTEvaluator<KeyT>>(threshold);
    }
    else if (expr_type == "range_and") {
        // col0 > low AND col0 < high
        KeyT low = mid - range;
        KeyT high = mid + range;
        return std::make_unique<RangeANDEvaluator<KeyT>>(low, high);
    }
    else if (expr_type == "two_col_sum") {
        // col0 + col1 > threshold
        KeyT threshold = static_cast<KeyT>(num_items * 2 * (1.0 - selectivity));
        return std::make_unique<TwoColSumEvaluator<KeyT>>(threshold);
    }
    else if (expr_type == "four_col_equal") {
        // (col0 + col1) == (col2 + col3)
        return std::make_unique<FourColEqualEvaluator<KeyT>>();
    }
    
    throw std::runtime_error("Unknown handwritten expression type: " + expr_type);
}

template<typename KeyT>
bool check_skip(select_handwritten_params const& params) {
    // No skipping for now since we only have two expressions
    return false;
}

// =============================================================================
// Benchmark Function
// =============================================================================

template<typename KeyT>
void bench_select_handwritten(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem(); // ensure clean state

    select_handwritten_params params(state);

    if (check_skip<KeyT>(params)) {
        state.skip("Skipped configuration");
        return;
    }

    gamax::size_type num_items = params.num_items;
    
    // Get the handwritten expression evaluator
    auto evaluator = get_handwritten_evaluator<KeyT>(params.expr_type, params.selectivity, num_items);
    int ncols = evaluator->num_columns();
    
    // Generate table with uniform distribution
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto table = gamax::datagen::device::pk_uniform_id(key_type, num_items, ncols, 42);
    
    // Allocate output buffer using gamax::column
    auto bool_type = gamax::data_type(gamax::type_to_id<bool>());
    auto output = gamax::make_column(bool_type, num_items, false);
    auto output_view = output->mutable_view();
    
    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.add_element_count(num_items, "NumElements");
    // state.collect_cupti_metrics();
    
    // Add expression info to summary
    nvbench::summary &summary = state.add_summary("expr_info");
    summary.set_string("name", "expr_info");
    summary.set_string("description", "The expression being evaluated");
    summary.set_string("value", evaluator->to_string());

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) { 
                evaluator->evaluate(table->view(), output_view, launch.get_stream());
                cudaStreamSynchronize(launch.get_stream());
            });
    }
    catch(const std::exception& e) {
        std::cerr << "Error in select_handwritten benchmark: " << e.what() << '\n';
        std::ofstream out("select_handwritten_error.txt", std::ios::app);
        out << "Expression: " << evaluator->to_string() << "\n";
        out << "Error: " << e.what() << "\n\n";
    }

    print_gpu_arr(output_view.data<bool>(), 1);
}

NVBENCH_BENCH_TYPES(bench_select_handwritten,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("select_handwritten")
.set_type_axes_names({"KeyType"})
.add_string_axis("expr_type", {
    "simple_gt",
    "range_and",
    "two_col_sum",
    "four_col_equal"
})
.add_int64_power_of_two_axis("num_items", nvbench::range(20, 27, 1))
.add_float64_axis("selectivity", {0.01, 0.1, 0.25, 0.5, 0.75, 0.9});
