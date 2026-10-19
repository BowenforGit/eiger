// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Benchmark for gamax::argsort_strings and gamax::sort_strings using nvbench.

#include <nvbench/nvbench.cuh>

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>

#include "data_gen/string_generator.hpp"
#include "string/string.hpp"
#include "utils/utils.cuh"

namespace {

struct bm_params {
    gamax::size_type num_strings;
    gamax::size_type string_length;
    gamax::string_impl_id impl;
    bool use_prefix_sort;
    std::string op;

    explicit bm_params(nvbench::state& state)
        : num_strings(state.get_int64("num_strings")),
          string_length(state.get_int64("string_length")),
          impl(gamax::string_impl_id::ARROW_STRING),
          use_prefix_sort(state.get_int64("use_prefix_sort") != 0),
          op(state.get_string("op")) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "num_strings: " << num_strings << ", "
            << "string_length: " << string_length << ", "
            << "use_prefix_sort: " << (use_prefix_sort ? "true" : "false") << ", "
            << "op: " << op;
        return oss.str();
    }
          
    bool should_skip(nvbench::state& state) {
        // Add any skip conditions here if needed
        return false;
    }
};

} // namespace

void bench_string_sort(nvbench::state& state) {
    gamax::prewarm_device_memory_resource();

    bm_params params(state);
    if (params.should_skip(state)) {
        return;
    }

    cudaStream_t stream = cudaStreamDefault;

    // Generate random strings for sorting
    auto strings = gamax::datagen::device::generate_random_strings(
        params.num_strings,
        params.string_length,
        stream,
        params.impl,
        42);  // seed

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.num_strings, "num_strings");
    state.add_global_memory_reads<char>(
        static_cast<int64_t>(params.num_strings) * params.string_length, 
        "InputBytes");

    gamax::string_op_options op_options;
    op_options.use_prefix_sort = params.use_prefix_sort;

    cudaDeviceSynchronize();
    std::cout << "String column generated: " << params.num_strings 
              << " strings, length=" << params.string_length << std::endl;

    try {
        if (params.op == "argsort") {
            state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
                auto indices = gamax::argsort_strings(
                    strings->view_s(), 
                    gamax::sort_order::ASCENDING,
                    launch.get_stream(),
                    op_options);
                cudaStreamSynchronize(launch.get_stream());
            });
        } else if (params.op == "sort") {
            state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
                auto sorted = gamax::sort_strings(
                    strings->view_s(), 
                    gamax::sort_order::ASCENDING,
                    launch.get_stream(),
                    op_options);
                cudaStreamSynchronize(launch.get_stream());
            });
        } else {
            throw std::runtime_error("Unknown operation: " + params.op);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << "Error: " << e.what() << "\n";
        out << "Params: " << params.to_string() << "\n\n";
    }
}

NVBENCH_BENCH(bench_string_sort)
    .set_name("string_sort")
    .add_string_axis("op", {"argsort", "sort"})
    .add_int64_axis("string_length", {16, 32, 64, 128, 256})
    .add_int64_power_of_two_axis("num_strings", {20, 22, 24, 26})
    .add_int64_axis("use_prefix_sort", {0, 1});
