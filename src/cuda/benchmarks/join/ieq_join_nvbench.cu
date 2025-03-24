#include <nvbench/nvbench.cuh>

#include <string>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>
#include <type_traits>

#ifdef __NVCC__
#include <nvtx3/nvtx3.hpp>
#endif

#include "operators/join/join.hpp"
#include "operators/join/inner_join.hpp"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"

struct join_params {
    std::string join_algo;
    std::string mat_algo;
    int64_t num_items;
    int64_t ratio;
    int64_t num_payloads_each;
    double match_ratio;
    double zipf_factor;

    join_params(nvbench::state& state) {
        join_algo = state.get_string("join_algo");
        mat_algo = state.get_string("mat_algo");
        num_items = state.get_int64("num_items");
        ratio = state.get_int64("ratio");
        num_payloads_each = state.get_int64("num_payloads_each");
        match_ratio = state.get_float64("match_ratio");
        zipf_factor = state.get_float64("zipf_factor");
    }

    gamax::join::join_hint hint() {
        gamax::join::join_hint hint;
        hint.max_out = num_items * ratio;
        hint.unique_left = true;
        hint.mat_algo = (mat_algo == "gfur") ? gamax::join::materialization_algo::GFUR : gamax::join::materialization_algo::GFTR;
        hint.join_algo = (join_algo == "smj") ? gamax::join::join_algo::SORT : 
                         (join_algo == "phj") ? gamax::join::join_algo::PARTITION : 
                         gamax::join::join_algo::HASH;
        return hint;
    }
};

template<typename KeyT>
bool check_skip(join_params const& params) {
    if(params.zipf_factor > 0.0) {
        return !(sizeof(KeyT) == 4 && params.match_ratio == 1.0 && params.num_payloads_each == 2 && params.ratio == 2);
    }

    if(params.match_ratio != 1.0) {
        return !(sizeof(KeyT) == 4 && params.zipf_factor == 0.0 && params.num_payloads_each == 2 && params.ratio == 2);
    }

    if(params.join_algo == "hash" && params.mat_algo != "gfur") {
        return true;
    }

    return false;
}

template<typename KeyT>
void bench_inner_eq_join(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem(); // make sure if a test fails, the following test will not be affected

    join_params params(state);

    if(check_skip<KeyT>(params)) {
        return;
    }

    gamax::size_type npk = params.num_items;
    gamax::size_type nfk = params.num_items * params.ratio;
    gamax::size_type ncols = params.num_payloads_each+1;

    auto hint = params.hint();
    
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto left = gamax::datagen::device::pk_uniform_id(key_type, npk, ncols, 42);
    auto right = gamax::datagen::device::fk_uniform_id(key_type, npk, nfk, ncols, params.match_ratio, 39);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.add_element_count(params.num_items*(params.ratio+1), "NumElements");

    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) { 
            auto res = gamax::join::inner_eq_join(left->view(), right->view(), hint, launch.get_stream()); 
            cudaStreamSynchronize(launch.get_stream());
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
    }
}

NVBENCH_BENCH_TYPES(bench_inner_eq_join,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("inner_eq_join")
.set_type_axes_names({"Key"})
.add_string_axis("join_algo", {"smj", "phj", "hash"})
.add_string_axis("mat_algo", {"gfur", "gftr"})
.add_int64_power_of_two_axis("num_items", {27, 28})
.add_int64_axis("ratio", {2})
.add_int64_axis("num_payloads_each", nvbench::range(1, 3, 1))
.add_float64_axis("match_ratio", {0.0625, 0.125, 0.25, 0.5, 1.0})
.add_float64_axis("zipf_factor", nvbench::range(0.0, 2.0, 0.5));