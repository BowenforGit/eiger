// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

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

#include "containers/multiset.cuh"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/sentinel.hpp"

// Compile-time CG sizes to sweep: powers of 2 in the practical range for cuco
using CGSizes = nvbench::enum_type_list<1, 2, 4>;

struct multiset_params {
    int64_t unique_keys;
    int64_t build_num_keys;
    int64_t probe_num_keys;
    int64_t build_dup;
    int64_t probe_dup;
    double hit_ratio;
    double load_factor;

    multiset_params(nvbench::state& state) {
        unique_keys = state.get_int64("unique_keys");
        build_dup = state.get_int64("build_dup");
        build_num_keys = unique_keys * build_dup;
        probe_dup = state.get_int64_or_default("probe_dup", -1);
        if(probe_dup == -1) {
            probe_num_keys = state.get_int64("probe_num_keys");
            probe_dup = probe_num_keys / unique_keys;
            assert(probe_dup >= 1);
        } else {
            probe_num_keys = unique_keys * probe_dup;
        }
        hit_ratio = state.get_float64("hit_ratio");
        load_factor = state.get_float64("load_factor");
    }
};

template<typename KeyT>
bool check_skip(multiset_params const& params) {
    // We want to build the hash table on the smaller side
    if (params.build_num_keys > params.probe_num_keys) {
        return true;
    }
    return false;
}

// =============================================================================
// Benchmark: insert_with_tids
// Measures the time to construct the multiset and insert build TIDs.
// A fresh multiset is created per iteration since insertion is a one-shot op.
// =============================================================================
template<typename KeyT, nvbench::int32_t CGSize>
void bench_multiset_insert(nvbench::state& state,
                           nvbench::type_list<KeyT, nvbench::enum_type<CGSize>>) {
    gamax::prewarm_device_memory_resource();

    multiset_params params(state);

    if(check_skip<KeyT>(params)) {
        return;
    }

    constexpr int ncols = 1;
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto build = gamax::datagen::device::fk_uniform_id(
        key_type, params.unique_keys, params.build_num_keys, ncols, 1.0, 42,
        cudaStreamDefault);

    cudaStream_t stream = 0;
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    {
        nvbench::summary &summary = state.add_summary("cg_size");
        summary.set_string("name", "cg_size");
        summary.set_string("description", "The size of the cooperative group for probing");
        summary.set_int64("value", static_cast<int64_t>(CGSize));
    }

    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                cudaStream_t stream = launch.get_stream();
                // Construct a fresh multiset each iteration (insert is one-shot)
                gamax::rmm_allocator_handle<gamax::size_type> allocator{stream};
                auto set = gamax::containers::multiset<
                    KeyT, gamax::size_type,
                    gamax::rmm_cuco_allocator<gamax::size_type>, CGSize>(
                    build->get_column(0).view(),
                    params.load_factor,
                    gamax::sentinel<gamax::size_type>(),
                    allocator.get(),
                    stream);
                set.insert_with_tids(stream);
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
    }
}

// =============================================================================
// Benchmark: count
// Measures the time to count matches for probe keys against a pre-built multiset.
// =============================================================================
template<typename KeyT, nvbench::int32_t CGSize>
void bench_multiset_count(nvbench::state& state,
                          nvbench::type_list<KeyT, nvbench::enum_type<CGSize>>) {
    gamax::prewarm_device_memory_resource();

    multiset_params params(state);

    if(check_skip<KeyT>(params)) {
        return;
    }

    constexpr int ncols = 1;
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto build = gamax::datagen::device::fk_uniform_id(
        key_type, params.unique_keys, params.build_num_keys, ncols, 1.0, 42,
        cudaStreamDefault);
    auto probe = gamax::datagen::device::fk_uniform_id(
        key_type, params.unique_keys, params.probe_num_keys, ncols, params.hit_ratio, 39,
        cudaStreamDefault);

    cudaStream_t stream = 0;

    gamax::rmm_allocator_handle<gamax::size_type> allocator{stream};
    auto set = gamax::containers::multiset<
        KeyT, gamax::size_type,
        gamax::rmm_cuco_allocator<gamax::size_type>, CGSize>(
        build->get_column(0).view(),
        params.load_factor,
        gamax::sentinel<gamax::size_type>(),
        allocator.get(),
        stream);

    set.insert_with_tids(stream);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    {
        nvbench::summary &summary = state.add_summary("cg_size");
        summary.set_string("name", "cg_size");
        summary.set_string("description", "The size of the cooperative group for probing");
        summary.set_int64("value", static_cast<int64_t>(CGSize));
    }

    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                cudaStream_t stream = launch.get_stream();
                auto n_matches = set.count(probe->get_column(0).view(), stream);
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
    }
}

// =============================================================================
// Benchmark: probe
// Measures the time to retrieve matching (build_tid, probe_tid) pairs.
// The output_size is pre-computed via count() outside the timed region.
// =============================================================================
template<typename KeyT, nvbench::int32_t CGSize>
void bench_multiset_probe(nvbench::state& state,
                          nvbench::type_list<KeyT, nvbench::enum_type<CGSize>>) {
    gamax::prewarm_device_memory_resource();

    multiset_params params(state);

    if(check_skip<KeyT>(params)) {
        return;
    }

    constexpr int ncols = 1;
    auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto build = gamax::datagen::device::fk_uniform_id(
        key_type, params.unique_keys, params.build_num_keys, ncols, 1.0, 42,
        cudaStreamDefault);
    auto probe = gamax::datagen::device::fk_uniform_id(
        key_type, params.unique_keys, params.probe_num_keys, ncols, params.hit_ratio, 39,
        cudaStreamDefault);

    cudaStream_t stream = 0;

    gamax::rmm_allocator_handle<gamax::size_type> allocator{stream};
    auto set = gamax::containers::multiset<
        KeyT, gamax::size_type,
        gamax::rmm_cuco_allocator<gamax::size_type>, CGSize>(
        build->get_column(0).view(),
        params.load_factor,
        gamax::sentinel<gamax::size_type>(),
        allocator.get(),
        stream);

    set.insert_with_tids(stream);

    // Pre-compute output size (not part of the timed probe benchmark)
    auto output_size = static_cast<gamax::size_type>(
        set.count(probe->get_column(0).view(), stream));

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    {
        nvbench::summary &summary = state.add_summary("cg_size");
        summary.set_string("name", "cg_size");
        summary.set_string("description", "The size of the cooperative group for probing");
        summary.set_int64("value", static_cast<int64_t>(CGSize));
    }

    try
    {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                cudaStream_t stream = launch.get_stream();
                auto [build_tids, probe_tids] = set.probe(
                    probe->get_column(0).view(), output_size, stream);
        });
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::ofstream out("error.txt", std::ios::app);
        out << e.what() << "\n";
    }
}

// =============================================================================
// NVBench Registrations
// =============================================================================

// --- insert_with_tids ---
NVBENCH_BENCH_TYPES(bench_multiset_insert,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<int32_t, int64_t>,
        CGSizes
    ))
.set_name("multiset_insert")
.set_type_axes_names({"Key", "CGSize"})
.add_int64_power_of_two_axis("unique_keys", nvbench::range(10, 27))
.add_int64_axis("build_dup", {1, 2, 4})
.add_int64_power_of_two_axis("probe_num_keys", {29}) // ignored
.add_float64_axis("hit_ratio", {1.0})
.add_float64_axis("load_factor", {0.25, 0.5});

// --- count (build sweep, no explicit probe_dup) ---
NVBENCH_BENCH_TYPES(bench_multiset_count,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<int32_t, int64_t>,
        CGSizes
    ))
.set_name("multiset_count_build_sweep")
.set_type_axes_names({"Key", "CGSize"})
.add_int64_power_of_two_axis("unique_keys", nvbench::range(10, 27))
.add_int64_axis("build_dup", {1, 2, 4})
.add_int64_power_of_two_axis("probe_num_keys", {29})
.add_float64_axis("hit_ratio", {0.125, 0.25, 0.5, 1.0})
.add_float64_axis("load_factor", {0.25, 0.5});

// --- probe ---
NVBENCH_BENCH_TYPES(bench_multiset_probe,
    NVBENCH_TYPE_AXES(
        nvbench::type_list<int32_t, int64_t>,
        CGSizes
    ))
.set_name("multiset_probe_build_sweep")
.set_type_axes_names({"Key", "CGSize"})
.add_int64_power_of_two_axis("unique_keys", nvbench::range(10, 27))
.add_int64_axis("build_dup", {1, 2, 4})
.add_int64_power_of_two_axis("probe_num_keys", {29})
.add_float64_axis("hit_ratio", {0.125, 0.25, 0.5, 1.0})
.add_float64_axis("load_factor", {0.25, 0.5});
