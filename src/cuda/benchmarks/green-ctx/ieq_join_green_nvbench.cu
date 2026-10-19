// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Green-context SM-scaling variant of join/ieq_join_nvbench.cu.
// Focus: vary_left_table_size, left_ncols=right_ncols=2, mat_algo=gfur,
// join_algo in {smj, hash, phj}. Adds a `green_sms` axis: the timed join runs on
// a CUDA green-context SM partition of that many SMs (0 = native full device).
// A cudaDeviceSynchronize() orders datagen (default stream) before the timed join
// on the green stream (which does NOT serialize with stream 0). See green_ctx.hpp.
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

#include "../common/nvtx_stub.hpp"
#include "green_ctx.hpp"

#include "operators/join/join.hpp"
#include "operators/join/inner_join.hpp"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"

struct join_params {
    std::string join_algo;
    std::string mat_algo;
    int64_t left_num_items;
    int64_t right_num_items;
    int64_t left_ncols;
    int64_t right_ncols;
    double match_ratio;
    double zipf_factor;

    join_params(nvbench::state& state) {
        join_algo = state.get_string("join_algo");
        mat_algo = state.get_string("mat_algo");
        left_num_items = state.get_int64("left_num_items");
        right_num_items = state.get_int64_or_default("right_num_items", -1);
        auto ratio = state.get_int64_or_default("ratio", -1);
        if(right_num_items == -1 && ratio != -1) {
            right_num_items = left_num_items * ratio;
        } else if(right_num_items == -1 && ratio == -1) {
            throw std::runtime_error("Either right_num_items or ratio must be set.");
        }
        match_ratio = state.get_float64("match_ratio");
        zipf_factor = state.get_float64("zipf_factor");
        left_ncols = state.get_int64("left_ncols");
        right_ncols = state.get_int64("right_ncols");
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "join_algo: " << join_algo << ", "
            << "mat_algo: " << mat_algo << ", "
            << "left_num_items: " << left_num_items << ", "
            << "right_num_items: " << right_num_items << ", "
            << "left_ncols: " << left_ncols << ", "
            << "right_ncols: " << right_ncols << ", "
            << "match_ratio: " << match_ratio << ", "
            << "zipf_factor: " << zipf_factor;
        return oss.str();
    }

    gamax::join::join_hint hint() {
        gamax::join::join_hint hint;
        hint.max_out = right_num_items;
        hint.unique_left = true;
        hint.mat_algo = (mat_algo == "gfur") ?
                        gamax::join::materialization_algo::GFUR :
                        gamax::join::materialization_algo::GFTR;
        hint.join_algo = (join_algo == "smj") ? gamax::join::join_algo::SORT :
                         (join_algo == "phj") ? gamax::join::join_algo::PARTITION :
                         gamax::join::join_algo::HASH;
        return hint;
    }
};

template<typename KeyT>
bool check_skip(join_params const& params) {
    if(params.join_algo == "hash" && params.mat_algo != "gfur") {
        return true;
    }
    if(params.left_num_items > params.right_num_items) {
        return true;
    }
    return false;
}

template<typename KeyT>
void bench_inner_eq_join(nvbench::state& state, nvbench::type_list<KeyT>) {
    gamax::prewarm_device_memory_resource();

    join_params params(state);
    if(check_skip<KeyT>(params)) return;

    gamax::size_type npk = params.left_num_items;
    gamax::size_type nfk = params.right_num_items;

    auto hint = params.hint();

    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto left = gamax::datagen::device::pk_uniform_id(
        key_type, npk, params.left_ncols, 42, cudaStreamDefault);
    std::unique_ptr<gamax::table> right =
        (params.zipf_factor == 0.0) ?
        gamax::datagen::device::fk_uniform_id(
            key_type, npk, nfk, params.right_ncols, params.match_ratio, 39,
            cudaStreamDefault) :
        gamax::datagen::device::fk_zipf_id(
            key_type, npk, nfk, params.right_ncols, params.zipf_factor, 39,
            cudaStreamDefault);

    GreenCtx green;
    cudaStream_t bench_stream = green_bench_stream(state, green);

    cudaDeviceSynchronize();  // finish datagen before timed join on the green stream
    state.set_cuda_stream(nvbench::make_cuda_stream_view(bench_stream));
    state.add_element_count(params.left_num_items+params.right_num_items, "NumElements");

    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
            cudaStream_t stream = launch.get_stream();
            nvtxRangePushA("inner_eq_join");
            auto res = gamax::join::inner_eq_join(left->view(), right->view(), hint, stream);
            cudaStreamSynchronize(stream);
            nvtxRangePop();
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << "Error: " << e.what() << "\n";
        out << "Params: " << params.to_string() << "\n\n";
    }
    green_teardown(green);
}

// User-fixed: vary_left_table_size, ncols=2/2, mat=gfur, join_algo in {smj,hash,phj},
// right=2^28. Swept from the CLI: green_sms, left_num_items, join_algo, Key.
NVBENCH_BENCH_TYPES(bench_inner_eq_join,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("vary_left_table_size")
.set_type_axes_names({"Key"})
.add_string_axis("join_algo", {"smj", "hash", "phj"})
.add_string_axis("mat_algo", {"gfur"})
.add_int64_power_of_two_axis("left_num_items", {24})
.add_int64_power_of_two_axis("right_num_items", {28})
.add_int64_axis("left_ncols", {2})
.add_int64_axis("right_ncols", {2})
.add_float64_axis("match_ratio", {1.0})
.add_float64_axis("zipf_factor", {0.0})
.add_int64_axis("green_sms", {0});
