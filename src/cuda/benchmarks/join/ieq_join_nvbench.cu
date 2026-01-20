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
    release_all_mem();

    join_params params(state);

    if(check_skip<KeyT>(params)) {
        return;
    }

    gamax::size_type npk = params.left_num_items;
    gamax::size_type nfk = params.right_num_items;

    auto hint = params.hint();
    
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto left = gamax::datagen::device::pk_uniform_id(key_type, npk, params.left_ncols, 42);
    std::unique_ptr<gamax::table> right = 
        (params.zipf_factor == 0.0) ? 
        gamax::datagen::device::fk_uniform_id(key_type, npk, nfk, params.right_ncols, params.match_ratio, 39) :
        gamax::datagen::device::fk_zipf_id(key_type, npk, nfk, params.right_ncols, params.zipf_factor, 39);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.add_element_count(params.left_num_items+params.right_num_items, "NumElements");

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
        out << "Error: " << e.what() << "\n";
        out << "Params: " << params.to_string() << "\n\n";
    }
}

std::vector<std::string> JOIN_ALGO_RANGE = {"smj", "phj", "hash"};
std::vector<std::string> MAT_ALGO_RANGE = {"gfur", "gftr"};
auto NUM_ITEMS_RANGE = nvbench::range(10, 28, 1);
std::vector<int64_t> NCOLS_RANGE = {2, 4};
auto DEFAULT_NCOLS = std::vector<int64_t> {4};
std::vector<double> DEFAULT_MATCH_RATIO = {1.0};
auto MATCH_RATIO_RANGE = nvbench::range(0.1, 1.0, 0.1);
std::vector<double> DEFAULT_ZIPF_FACTOR = {0.0};
auto ZIPF_FACTOR_RANGE = nvbench::range(0.5, 2.0, 0.5);
std::vector<int64_t> DEFAULT_RIGHT_NUM_ITEMS = {28};

NVBENCH_BENCH_TYPES(bench_inner_eq_join,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("vary_left_table_size")
.set_type_axes_names({"Key"})
.add_string_axis("join_algo", JOIN_ALGO_RANGE)
.add_string_axis("mat_algo", MAT_ALGO_RANGE)
.add_int64_power_of_two_axis("left_num_items", nvbench::range(10, 28, 1))
.add_int64_power_of_two_axis("right_num_items", DEFAULT_RIGHT_NUM_ITEMS)
.add_int64_axis("left_ncols", NCOLS_RANGE)
.add_int64_axis("right_ncols", NCOLS_RANGE)
.add_float64_axis("match_ratio", DEFAULT_MATCH_RATIO)
.add_float64_axis("zipf_factor", DEFAULT_ZIPF_FACTOR);

NVBENCH_BENCH_TYPES(bench_inner_eq_join,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("vary_right_table_size")
.set_type_axes_names({"Key"})
.add_string_axis("join_algo", JOIN_ALGO_RANGE)
.add_string_axis("mat_algo", MAT_ALGO_RANGE)
.add_int64_power_of_two_axis("left_num_items", nvbench::range(12, 24, 4))
.add_int64_power_of_two_axis("right_num_items", nvbench::range(12, 28, 1))
.add_int64_axis("left_ncols", NCOLS_RANGE)
.add_int64_axis("right_ncols", NCOLS_RANGE)
.add_float64_axis("match_ratio", DEFAULT_MATCH_RATIO)
.add_float64_axis("zipf_factor", DEFAULT_ZIPF_FACTOR);

NVBENCH_BENCH_TYPES(bench_inner_eq_join,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("vary_match_ratio")
.set_type_axes_names({"Key"})
.add_string_axis("join_algo", JOIN_ALGO_RANGE)
.add_string_axis("mat_algo", MAT_ALGO_RANGE)
.add_int64_power_of_two_axis("left_num_items", nvbench::range(12, 28, 4))
.add_int64_power_of_two_axis("right_num_items", DEFAULT_RIGHT_NUM_ITEMS)
.add_int64_axis("left_ncols", DEFAULT_NCOLS)
.add_int64_axis("right_ncols", DEFAULT_NCOLS)
.add_float64_axis("match_ratio", MATCH_RATIO_RANGE)
.add_float64_axis("zipf_factor", DEFAULT_ZIPF_FACTOR);

NVBENCH_BENCH_TYPES(bench_inner_eq_join,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
.set_name("vary_zipfian_factor")
.set_type_axes_names({"Key"})
.add_string_axis("join_algo", JOIN_ALGO_RANGE)
.add_string_axis("mat_algo", MAT_ALGO_RANGE)
.add_int64_power_of_two_axis("left_num_items", nvbench::range(12, 28, 4))
.add_int64_power_of_two_axis("right_num_items", DEFAULT_RIGHT_NUM_ITEMS)
.add_int64_axis("left_ncols", DEFAULT_NCOLS)
.add_int64_axis("right_ncols", DEFAULT_NCOLS)
.add_float64_axis("match_ratio", DEFAULT_MATCH_RATIO)
.add_float64_axis("zipf_factor", ZIPF_FACTOR_RANGE);