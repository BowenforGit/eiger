#include <nvbench/nvbench.cuh>

#include <string>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>
#include <type_traits>
#include <cassert>

#include "../common/nvtx_stub.hpp"

#include "containers/map.cuh"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/sentinel.hpp"

constexpr int CG_SIZE = 2;

struct map_params {
    int64_t unique_keys;
    int64_t build_num_keys;
    int64_t probe_num_keys;
    int64_t probe_dup;
    double hit_ratio;
    double load_factor;

    map_params(nvbench::state& state) {
        unique_keys = state.get_int64("unique_keys");
        build_num_keys = unique_keys;
        probe_dup = state.get_int64_or_default("probe_dup", -1);
        if(probe_dup == -1) {
            probe_num_keys = state.get_int64_or_default("probe_num_keys", (1 << 29));
            probe_dup = probe_num_keys / unique_keys;
            assert(probe_dup >= 1);
        } else {
            probe_num_keys = unique_keys * probe_dup;
        }
        hit_ratio = state.get_float64_or_default("hit_ratio", 1.0);
        load_factor = state.get_float64("load_factor");
    }
};

template<typename KeyT>
bool check_skip(map_params const& params) {
    // We also want to build the hash table on the smaller side
    if (params.build_num_keys > params.probe_num_keys) {
        return true;
    }
    return false;
}

template<typename KeyT>
void bench_map_count(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem(); // make sure if a test fails, the following test will not be affected

    using value_t = KeyT;

    map_params params(state);

    if(check_skip<KeyT>(params)) {
        return;
    }
    
    constexpr int ncols = 1;
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto build = gamax::datagen::device::fk_uniform_id(key_type, params.unique_keys, params.build_num_keys, ncols, 1.0, 42);
    auto probe = gamax::datagen::device::fk_uniform_id(key_type, params.unique_keys, params.probe_num_keys, ncols, params.hit_ratio, 39);

    cudaStream_t stream = 0;

    auto map = gamax::containers::map<KeyT, value_t, CG_SIZE>(
            build->num_rows(),
            params.load_factor,
            gamax::sentinel<KeyT>(),
            gamax::sentinel<value_t>(),
            *get_thrust_allocator<cuco::pair<KeyT, value_t>>(stream),
            stream);
    
    map.insert_with_tids(build->get_column(0).view(), stream);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

     {
        nvbench::summary &summary = state.add_summary("cg_size");
        summary.set_string("name", "cg_size");
        summary.set_string("description", "The size of the cooperative group for probing");
        summary.set_int64("value", static_cast<int64_t>(map.cg_size()));
    }

    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                cudaStream_t stream = launch.get_stream();
                auto n_matches = map.count(probe->get_column(0).view(), stream);
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
    }
}

template<typename KeyT>
void bench_map_insert(nvbench::state& state, nvbench::type_list<KeyT>) {
    release_all_mem(); // make sure if a test fails, the following test will not be affected

    using value_t = KeyT;

    map_params params(state);

    if(check_skip<KeyT>(params)) {
        return;
    }
    
    constexpr int ncols = 1;
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto build = gamax::datagen::device::fk_uniform_id(key_type, params.unique_keys, params.build_num_keys, ncols, 1.0, 42);

    cudaStream_t stream = 0;

    auto map = gamax::containers::map<KeyT, value_t, CG_SIZE>(
            build->num_rows(),
            params.load_factor,
            gamax::sentinel<KeyT>(),
            gamax::sentinel<value_t>(),
            *get_thrust_allocator<cuco::pair<KeyT, value_t>>(stream),
            stream);
    
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

     {
        nvbench::summary &summary = state.add_summary("cg_size");
        summary.set_string("name", "cg_size");
        summary.set_string("description", "The size of the cooperative group for probing");
        summary.set_int64("value", static_cast<int64_t>(map.cg_size()));
    }

    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                cudaStream_t stream = launch.get_stream();
                map.insert_with_tids(build->get_column(0).view(), stream);
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
    }
}

NVBENCH_BENCH_TYPES(bench_map_count,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<int32_t, int64_t>
    ))
.set_name("bench_map_count_build_sweep")
.set_type_axes_names({"Key"})
.add_int64_power_of_two_axis("unique_keys", nvbench::range(5, 27))
.add_int64_power_of_two_axis("probe_num_keys", {29})
.add_float64_axis("hit_ratio", {0.0625, 0.125, 0.25, 0.5, 1.0})
.add_float64_axis("load_factor", {0.125, 0.25, 0.5});

NVBENCH_BENCH_TYPES(bench_map_insert,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<int32_t, int64_t>
    ))
.set_name("bench_map_insert")
.set_type_axes_names({"Key"})
.add_int64_power_of_two_axis("unique_keys", nvbench::range(5, 29))
.add_float64_axis("load_factor", {0.125, 0.25, 0.5});