/**
 * @file stats_nvbench.cu
 * @brief NVBench benchmark for gamax::stats statistics computation functions.
 * 
 * Dimensions explored:
 * - Column sizes: varying powers of two
 * - Data types: int32_t, int64_t
 * - Statistics types: MIN, MAX, AVG, NDV_APPROX, INV_COUNT_APPROX
 * 
 * Histogram statistics (EQUI_WIDTH_HISTOGRAM, EQUI_DEPTH_HISTOGRAM) are 
 * currently not supported and are skipped.
 */

#include <nvbench/nvbench.cuh>

#include <string>
#include <iostream>
#include <memory>
#include <exception>
#include <fstream>

#include "../common/nvtx_stub.hpp"

#include "stats/stats_compute.hpp"
#include "stats/stats_storage.hpp"
#include "data_gen/table_gen.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"

namespace {

/**
 * @brief Convert a string to StatsType enum.
 */
gamax::stats::StatsType string_to_stats_type(const std::string& s) {
    if (s == "MIN") return gamax::stats::StatsType::MIN;
    if (s == "MAX") return gamax::stats::StatsType::MAX;
    if (s == "AVG") return gamax::stats::StatsType::AVG;
    if (s == "NDV_APPROX") return gamax::stats::StatsType::NDV_APPROX;
    if (s == "INV_COUNT_APPROX") return gamax::stats::StatsType::INV_COUNT_APPROX;
    if (s == "EQUI_WIDTH_HISTOGRAM") return gamax::stats::StatsType::EQUI_WIDTH_HISTOGRAM;
    if (s == "EQUI_DEPTH_HISTOGRAM") return gamax::stats::StatsType::EQUI_DEPTH_HISTOGRAM;
    return gamax::stats::StatsType::UNSUPPORTED_STATS_TYPE;
}

/**
 * @brief Check if the stats type is a histogram type.
 */
bool is_histogram_type(gamax::stats::StatsType type) {
    return type == gamax::stats::StatsType::EQUI_WIDTH_HISTOGRAM ||
           type == gamax::stats::StatsType::EQUI_DEPTH_HISTOGRAM;
}

} // anonymous namespace

// ============================================================================
// Benchmark: get_or_compute_scalar_stat
// ============================================================================

template<typename ValueT>
void bench_get_or_compute_scalar_stat(nvbench::state& state, nvbench::type_list<ValueT>) {
    // Memory cleanup to ensure consistent state
    release_all_mem();
    
    // Clear any cached stats from previous runs
    gamax::stats::StatsStorage::instance().clear_all();

    const gamax::size_type num_items = state.get_int64("num_items");
    const std::string stats_type_str = state.get_string("stats_type");
    constexpr int seed = 42;

    auto stats_type = string_to_stats_type(stats_type_str);
    
    // Skip histogram types - not supported yet
    if (is_histogram_type(stats_type)) {
        state.skip("Histogram statistics not supported yet.");
        return;
    }
    
    // Skip unsupported stats types
    if (stats_type == gamax::stats::StatsType::UNSUPPORTED_STATS_TYPE) {
        state.skip("Unsupported statistics type.");
        return;
    }

    // Generate random input column with uniformly distributed values
    auto value_type = gamax::data_type(gamax::type_to_id<ValueT>());
    auto input_table = gamax::datagen::device::fk_uniform_id(
        value_type, num_items, num_items, 1, 1.0, seed);
    auto input_col = input_table->get_column(0).view();

    cudaDeviceSynchronize();

    // Set up benchmark stream
    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));

    // Add throughput metrics
    state.add_element_count(num_items, "NumElements");
    state.add_global_memory_reads<ValueT>(num_items, "InputRead");

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                // Clear cache before each iteration to benchmark actual computation
                gamax::stats::StatsStorage::instance().clear_all();
                
                auto result = gamax::stats::get_or_compute_scalar_stat(
                    input_col,
                    stats_type,
                    launch.get_stream());
                cudaStreamSynchronize(launch.get_stream());
            });
    } catch (const std::exception& e) {
        std::cerr << "Scalar stat benchmark error: " << e.what() << std::endl;
        std::ofstream out("error.txt", std::ios::app);
        out << "bench_get_or_compute_scalar_stat error: " << e.what() << "\n";
        out << "Params: num_items=" << num_items << ", stats_type=" << stats_type_str << "\n\n";
        state.skip(e.what());
    }
}

// ============================================================================
// NVBench Registration
// ============================================================================

// Benchmark scalar stat computation (cold cache)
NVBENCH_BENCH_TYPES(bench_get_or_compute_scalar_stat,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("bench_scalar_stat_compute")
.set_type_axes_names({"ValueType"})
.add_int64_power_of_two_axis("num_items", nvbench::range(20, 28, 2))
.add_string_axis("stats_type", {"MIN", "MAX", "AVG", "NDV_APPROX"});
