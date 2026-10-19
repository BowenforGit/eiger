// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Benchmark for gamax::startswith using nvbench.

#include <nvbench/nvbench.cuh>

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <functional>
#include <sstream>

#include "data_gen/string_generator.hpp"
#include "string/string.hpp"
#include "utils/utils.cuh"
#include "utils/transform.hpp"

namespace {

using gamax::datagen::device::string_generation_options;
using gamax::datagen::device::string_length_distribution;

string_length_distribution parse_length_dist(const std::string& value) {
    if (value == "fixed") return string_length_distribution::FIXED;
    if (value == "uniform") return string_length_distribution::UNIFORM;
    if (value == "zipfian") return string_length_distribution::ZIPFIAN;
    throw std::runtime_error("Unknown length_dist value: " + value);
}

gamax::string_impl_id parse_impl(const std::string& value) {
    // if (value == "german") return gamax::string_impl_id::GERMAN_STRING;
    if (value == "arrow") return gamax::string_impl_id::ARROW_STRING;
    throw std::runtime_error("Unknown string impl value: " + value);
}

struct bm_params {
    gamax::size_type num_strings;
    string_length_distribution length_dist;
    gamax::string_impl_id impl;
    double selectivity;
    gamax::size_type fixed_length;
    gamax::size_type pattern_length;
    std::string op;
    int cg_size;
    bool use_kmp;
    int access_width;

    explicit bm_params(nvbench::state& state)
        : num_strings(state.get_int64("num_strings")),
          length_dist(parse_length_dist(state.get_string("length_dist"))),
          impl(parse_impl(state.get_string("impl"))),
          selectivity(state.get_float64("selectivity")),
          fixed_length(state.get_int64("fixed_length")),
          pattern_length(state.get_int64_or_default("pattern_length", state.get_int64("fixed_length"))),
          op(state.get_string("op")),
          cg_size(state.get_int64("cg_size")),
          use_kmp(state.get_int64_or_default("use_kmp", 0)),
          access_width(state.get_int64("access_width")) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "num_strings: " << num_strings << ", "
            << "fixed_length: " << fixed_length << ", "
            << "pattern_length: " << pattern_length << ", "
            << "op: " << op << ", "
            << "cg_size: " << cg_size << ", "
            << "use_kmp: " << use_kmp << ", "
            << "access_width: " << access_width << ", "
            << "selectivity: " << selectivity;
        return oss.str();
    }
          
    bool should_skip(nvbench::state& state) {
        if (length_dist != string_length_distribution::FIXED) {
            state.skip("Only FIXED length distribution is implemented in the generator.");
            return true;
        }
        if (selectivity != 1.0) {
            state.skip("Partial selectivity is not implemented in the generator.");
            return true;
        }
		if(pattern_length > fixed_length) {
			state.skip("pattern_length cannot be greater than fixed_length.");
			return true;
		}
		if(cg_size > pattern_length) {
			state.skip("cg_size must be less than or equal to pattern_length.");
			return true;
		}
        if(op != "contains" && use_kmp) {
            state.skip("KMP is only implemented for 'contains' operation.");
            return true;
        }
        if(op == "contains" && cg_size != 1) {
            state.skip("cg_size > 1 is not implemented for 'contains' operation.");
            return true;
        }
        if(use_kmp && access_width != 1) {
            state.skip("access_width > 1 is not implemented for KMP.");
            return true;
        }

        return false;
    }
};
} // namespace

void bench_string_ops(nvbench::state& state) {
    gamax::prewarm_device_memory_resource();

    bm_params params(state);
    if (params.should_skip(state)) {
        return;
    }

    string_generation_options options{};
    options.length_dist = params.length_dist;
    options.impl = params.impl;
    options.selectivity = params.selectivity;
    options.fixed.length = params.fixed_length;
    if(params.op == "startswith" or params.op == "equal") {
        options.pattern_pos = gamax::datagen::device::PREFIX;
    } else if(params.op == "endswith") {
        options.pattern_pos = gamax::datagen::device::SUFFIX;
    } else if(params.op == "contains") {
        options.pattern_pos = gamax::datagen::device::SUBSTRING;
    }

	std::string pattern(params.pattern_length, 'a');
	for (size_t i = 0; i < params.pattern_length; ++i) {
		pattern[i] = 'a' + (i % 26);
	}

    cudaStream_t stream = cudaStreamDefault;

    auto strings = gamax::datagen::device::generate_strings(
        params.num_strings, pattern, options, stream);
    auto pattern_scalar = gamax::make_string_scalar(pattern, stream);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.num_strings, "num_strings");

    gamax::string_op_options op_options;
    op_options.cg_size = params.cg_size;
    op_options.use_kmp = params.use_kmp;
    op_options.access_width = params.access_width;

    auto fn = gamax::startswith;
    if (params.op == "endswith") {
        fn = gamax::endswith;
    } else if (params.op == "contains") {
        fn = gamax::contains;
    } else if (params.op == "equal") {
        fn = gamax::equal;
    } else if (params.op != "startswith") {
        throw std::runtime_error("Unknown operation: " + params.op);
    }

    cudaDeviceSynchronize();
    std::cout << "string column is generated" << std::endl;

    try {
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
            auto out = fn(strings->view_s(), pattern_scalar->view(), launch.get_stream(), op_options);
            cudaStreamSynchronize(launch.get_stream());
        });
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << "Error: " << e.what() << "\n";
        out << "Params: " << params.to_string() << "\n\n";
    }

    auto out = fn(strings->view_s(), pattern_scalar->view(), stream, op_options);
    cudaStreamSynchronize(stream);
	assert(gamax::util::device::count_nonzero(out->view(), stream) == params.num_strings);
}

NVBENCH_BENCH(bench_string_ops)
    .set_name("string_partial_match")
    .add_string_axis("op", {"startswith", "endswith", "contains"})
    .add_string_axis("length_dist", {"fixed"})
    .add_string_axis("impl", {"arrow"})
    .add_float64_axis("selectivity", {1.0})
    .add_int64_axis("fixed_length", {30, 45, 60, 75, 90, 105, 120, 135})
    .add_int64_axis("pattern_length", {5, 50})
    .add_int64_power_of_two_axis("num_strings", {27})
    .add_int64_axis("cg_size", {1, 2, 4})
    .add_int64_axis("access_width", {1, 2, 4, 8})
    .add_int64_axis("use_kmp", {0, 1});

NVBENCH_BENCH(bench_string_ops)
    .set_name("string_full_match")
    .add_string_axis("op", {"equal"})
    .add_string_axis("length_dist", {"fixed"})
    .add_string_axis("impl", {"arrow"})
    .add_float64_axis("selectivity", {1.0})
    .add_int64_axis("fixed_length", {30, 45, 60, 75, 90, 105, 120, 135})
    .add_int64_power_of_two_axis("num_strings", {27})
    .add_int64_axis("cg_size", {1, 2, 4})
    .add_int64_axis("access_width", {1, 2, 4, 8});

NVBENCH_BENCH(bench_string_ops)
    .set_name("string_partial_match_bad_align")
    .add_string_axis("op", {"startswith", "endswith", "contains"})
    .add_string_axis("length_dist", {"fixed"})
    .add_string_axis("impl", {"arrow"})
    .add_float64_axis("selectivity", {1.0})
    .add_int64_axis("fixed_length", {17, 34, 51, 68, 85, 102, 119, 136})
    .add_int64_axis("pattern_length", {5, 50})
    .add_int64_power_of_two_axis("num_strings", {27})
    .add_int64_axis("cg_size", {1, 2, 4})
    .add_int64_axis("access_width", {1, 2, 4, 8})
    .add_int64_axis("use_kmp", {0, 1});

NVBENCH_BENCH(bench_string_ops)
    .set_name("string_full_match_bad_align")
    .add_string_axis("op", {"equal"})
    .add_string_axis("length_dist", {"fixed"})
    .add_string_axis("impl", {"arrow"})
    .add_float64_axis("selectivity", {1.0})
    .add_int64_axis("fixed_length", {17, 34, 51, 68, 85, 102, 119, 136})
    .add_int64_power_of_two_axis("num_strings", {27})
    .add_int64_axis("cg_size", {1, 2, 4})
    .add_int64_axis("access_width", {1, 2, 4, 8});
