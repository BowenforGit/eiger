// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../common/nvtx_stub.hpp"

#include "containers/multiset.cuh"
#include "data_gen/table_gen.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/sentinel.hpp"
#include "utils/utils.cuh"

// Keep the same cooperative-group sweep as multiset_nvbench.cu so results from
// the two suites are directly comparable.
using CGSizes = nvbench::enum_type_list<1, 2, 4>;

namespace {

struct chunked_multiset_params {
    std::int64_t unique_keys;
    std::int64_t build_dup;
    std::int64_t build_num_keys;
    std::int64_t probe_num_keys;
    std::int64_t chunk_size;
    double hit_ratio;
    double load_factor;

    explicit chunked_multiset_params(nvbench::state& state)
        : unique_keys{state.get_int64("unique_keys")},
          build_dup{state.get_int64("build_dup")},
          probe_num_keys{state.get_int64("probe_num_keys")},
          chunk_size{state.get_int64("chunk_size")},
          hit_ratio{state.get_float64("hit_ratio")},
          load_factor{state.get_float64("load_factor")}
    {
        build_num_keys =
            unique_keys > 0 && build_dup > 0 &&
                    unique_keys <= std::numeric_limits<std::int64_t>::max() / build_dup
                ? unique_keys * build_dup
                : 0;
    }

    bool validate(nvbench::state& state, std::int64_t input_size) const
    {
        if (unique_keys <= 0) {
            state.skip("unique_keys must be positive");
            return false;
        }
        if (build_dup <= 0 || build_num_keys <= 0) {
            state.skip("build_dup must be positive and unique_keys * build_dup must fit in int64");
            return false;
        }
        if (probe_num_keys <= 0) {
            state.skip("probe_num_keys must be positive");
            return false;
        }
        if (chunk_size <= 0) {
            state.skip("chunk_size must be positive");
            return false;
        }
        if (input_size <= 0) {
            state.skip("chunked input size must be positive");
            return false;
        }
        if (hit_ratio < 0.0 || hit_ratio > 1.0) {
            state.skip("hit_ratio must be in [0, 1]");
            return false;
        }
        if (load_factor <= 0.0 || load_factor > 1.0) {
            state.skip("load_factor must be in (0, 1]");
            return false;
        }
        return true;
    }
};

struct chunk_range {
    gamax::size_type first;
    gamax::size_type size;
};

std::vector<chunk_range> make_chunk_ranges(std::int64_t input_size, std::int64_t chunk_size)
{
    std::vector<chunk_range> ranges;
    for (std::int64_t first = 0; first < input_size;) {
        auto const length = std::min(chunk_size, input_size - first);
        ranges.push_back(
            {static_cast<gamax::size_type>(first), static_cast<gamax::size_type>(length)});
        first += length;
    }
    return ranges;
}

template <typename KeyT>
gamax::column_view make_slice(
    gamax::column_view const& full, gamax::size_type first, gamax::size_type size)
{
    return gamax::column_view(full.data<KeyT>() + first, size, full.type());
}

void add_int_summary(
    nvbench::state& state,
    std::string const& name,
    std::string const& description,
    std::int64_t value)
{
    auto& summary = state.add_summary(name);
    summary.set_string("name", name);
    summary.set_string("description", description);
    summary.set_int64("value", value);
}

template <nvbench::int32_t CGSize>
void add_common_summaries(
    nvbench::state& state,
    chunked_multiset_params const& params,
    std::vector<chunk_range> const& ranges)
{
    add_int_summary(
        state, "cg_size", "The cooperative-group size used for probing", CGSize);
    add_int_summary(
        state, "build_num_keys", "Total number of rows inserted into the multiset",
        params.build_num_keys);
    add_int_summary(
        state, "chunk_count", "Number of chunks used to process the timed input",
        static_cast<std::int64_t>(ranges.size()));
    add_int_summary(
        state, "tail_chunk_size", "Number of rows in the final chunk",
        static_cast<std::int64_t>(ranges.back().size));
}

template <typename KeyT, nvbench::int32_t CGSize>
using benchmark_multiset = gamax::containers::multiset<
    KeyT,
    gamax::size_type,
    gamax::rmm_cuco_allocator<gamax::size_type>,
    CGSize>;

// Insert all build rows into one full-capacity multiset, issuing one ranged
// insert kernel per chunk. The final synchronization is required because the
// multiset is owned by the timed callback and must outlive all queued kernels.
template <typename KeyT, nvbench::int32_t CGSize>
void bench_chunked_multiset_insert(
    nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<CGSize>>)
{
    gamax::prewarm_device_memory_resource();
    chunked_multiset_params const params{state};
    if (!params.validate(state, params.build_num_keys)) { return; }

    auto const ranges = make_chunk_ranges(params.build_num_keys, params.chunk_size);
    auto const key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto build = gamax::datagen::device::fk_uniform_id(
        key_type,
        params.unique_keys,
        params.build_num_keys,
        1,
        1.0,
        42,
        cudaStreamDefault);

    cudaStream_t stream = cudaStreamDefault;
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.build_num_keys, "BuildRows");
    add_common_summaries<CGSize>(state, params, ranges);

    try {
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
            cudaStream_t launch_stream = launch.get_stream();
            gamax::rmm_allocator_handle<gamax::size_type> allocator{launch_stream};
            benchmark_multiset<KeyT, CGSize> set{
                build->get_column(0).view(),
                params.load_factor,
                gamax::sentinel<gamax::size_type>(),
                allocator.get(),
                launch_stream};

            for (auto const& range : ranges) {
                set.insert_with_tids_async(
                    range.first, range.first + range.size, launch_stream);
            }
            cudaStreamSynchronize(launch_stream);
        });
    } catch (std::exception const& e) {
        state.skip(e.what());
    }
}

template <typename KeyT, nvbench::int32_t CGSize>
void bench_chunked_multiset_count(
    nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<CGSize>>)
{
    gamax::prewarm_device_memory_resource();
    chunked_multiset_params const params{state};
    if (!params.validate(state, params.probe_num_keys)) { return; }

    auto const ranges = make_chunk_ranges(params.probe_num_keys, params.chunk_size);
    auto const key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto build = gamax::datagen::device::fk_uniform_id(
        key_type,
        params.unique_keys,
        params.build_num_keys,
        1,
        1.0,
        42,
        cudaStreamDefault);
    auto probe = gamax::datagen::device::fk_uniform_id(
        key_type,
        params.unique_keys,
        params.probe_num_keys,
        1,
        params.hit_ratio,
        39,
        cudaStreamDefault);

    cudaStream_t stream = cudaStreamDefault;
    gamax::rmm_allocator_handle<gamax::size_type> allocator{stream};
    benchmark_multiset<KeyT, CGSize> set{
        build->get_column(0).view(),
        params.load_factor,
        gamax::sentinel<gamax::size_type>(),
        allocator.get(),
        stream};
    set.insert_with_tids(stream);

    auto const probe_view = probe->get_column(0).view();
    auto const expected_matches = set.count(probe_view, stream);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.probe_num_keys, "ProbeRows");
    add_common_summaries<CGSize>(state, params, ranges);
    add_int_summary(
        state,
        "match_count",
        "Total number of build/probe matches across all chunks",
        static_cast<std::int64_t>(expected_matches));

    std::size_t measured_matches = 0;
    try {
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
            measured_matches = 0;
            for (auto const& range : ranges) {
                measured_matches += set.count(
                    make_slice<KeyT>(probe_view, range.first, range.size),
                    launch.get_stream());
            }
        });
        if (measured_matches != expected_matches) {
            state.skip("chunked count result differs from whole-column count");
        }
    } catch (std::exception const& e) {
        state.skip(e.what());
    }
}

template <typename KeyT, nvbench::int32_t CGSize>
void bench_chunked_multiset_retrieve(
    nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<CGSize>>)
{
    gamax::prewarm_device_memory_resource();
    chunked_multiset_params const params{state};
    if (!params.validate(state, params.probe_num_keys)) { return; }

    auto const ranges = make_chunk_ranges(params.probe_num_keys, params.chunk_size);
    auto const key_type = gamax::data_type(gamax::type_to_id<KeyT>());
    auto build = gamax::datagen::device::fk_uniform_id(
        key_type,
        params.unique_keys,
        params.build_num_keys,
        1,
        1.0,
        42,
        cudaStreamDefault);
    auto probe = gamax::datagen::device::fk_uniform_id(
        key_type,
        params.unique_keys,
        params.probe_num_keys,
        1,
        params.hit_ratio,
        39,
        cudaStreamDefault);

    cudaStream_t stream = cudaStreamDefault;
    gamax::rmm_allocator_handle<gamax::size_type> allocator{stream};
    benchmark_multiset<KeyT, CGSize> set{
        build->get_column(0).view(),
        params.load_factor,
        gamax::sentinel<gamax::size_type>(),
        allocator.get(),
        stream};
    set.insert_with_tids(stream);

    // probe() requires the output size up front. Precompute one size per chunk
    // so count work is not accidentally included in the retrieve measurement.
    auto const probe_view = probe->get_column(0).view();
    std::vector<gamax::size_type> output_sizes;
    output_sizes.reserve(ranges.size());
    std::size_t expected_matches = 0;
    for (auto const& range : ranges) {
        auto const count = set.count(
            make_slice<KeyT>(probe_view, range.first, range.size), stream);
        if (count > static_cast<std::size_t>(std::numeric_limits<gamax::size_type>::max())) {
            state.skip("retrieve output size does not fit in gamax::size_type");
            return;
        }
        output_sizes.push_back(static_cast<gamax::size_type>(count));
        expected_matches += count;
    }

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.add_element_count(params.probe_num_keys, "ProbeRows");
    add_common_summaries<CGSize>(state, params, ranges);
    add_int_summary(
        state,
        "match_count",
        "Total number of build/probe pairs materialized across all chunks",
        static_cast<std::int64_t>(expected_matches));

    std::size_t retrieved_matches = 0;
    try {
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
            retrieved_matches = 0;
            for (std::size_t i = 0; i < ranges.size(); ++i) {
                auto const& range = ranges[i];
                auto [build_tids, probe_tids] = set.probe(
                    make_slice<KeyT>(probe_view, range.first, range.size),
                    output_sizes[i],
                    launch.get_stream());
                assert(build_tids->size() == probe_tids->size());
                retrieved_matches += static_cast<std::size_t>(build_tids->size());
            }
        });
        if (retrieved_matches != expected_matches) {
            state.skip("chunked retrieve result differs from precomputed match count");
        }
    } catch (std::exception const& e) {
        state.skip(e.what());
    }
}

}  // namespace

// The default cells are deliberately centered on the proposed experiment:
// 100M timed rows with 1--100 chunks. The remaining parameters are explicit
// axes so hit ratio, key width, and probing geometry can be filtered
// independently from the chunk-size sweep. Insert keeps build_dup=1 so the
// 100M chunk is a true one-kernel baseline; count/retrieve sweep build_dup
// because build-side multiplicity intentionally controls their output size.
NVBENCH_BENCH_TYPES(
    bench_chunked_multiset_insert,
    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t, std::int64_t>, CGSizes))
    .set_name("multiset_chunked_insert")
    .set_type_axes_names({"Key", "CGSize"})
    .add_int64_axis("unique_keys", {100'000'000})
    .add_int64_axis("build_dup", {1})
    .add_int64_axis("probe_num_keys", {100'000'000})  // unused by insert
    .add_int64_axis("chunk_size", {100'000'000, 50'000'000, 20'000'000, 10'000'000,
                                    5'000'000, 1'000'000})
    .add_float64_axis("hit_ratio", {1.0})  // unused by insert
    .add_float64_axis("load_factor", {0.5});

NVBENCH_BENCH_TYPES(
    bench_chunked_multiset_count,
    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t, std::int64_t>, CGSizes))
    .set_name("multiset_chunked_count")
    .set_type_axes_names({"Key", "CGSize"})
    .add_int64_axis("unique_keys", {10'000'000})
    .add_int64_axis("build_dup", {1, 2, 4})
    .add_int64_axis("probe_num_keys", {100'000'000})
    .add_int64_axis("chunk_size", {100'000'000, 50'000'000, 20'000'000, 10'000'000,
                                    5'000'000, 1'000'000})
    .add_float64_axis("hit_ratio", {0.5, 1.0})
    .add_float64_axis("load_factor", {0.5});

NVBENCH_BENCH_TYPES(
    bench_chunked_multiset_retrieve,
    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t, std::int64_t>, CGSizes))
    .set_name("multiset_chunked_retrieve")
    .set_type_axes_names({"Key", "CGSize"})
    .add_int64_axis("unique_keys", {10'000'000})
    .add_int64_axis("build_dup", {1, 2, 4})
    .add_int64_axis("probe_num_keys", {100'000'000})
    .add_int64_axis("chunk_size", {100'000'000, 50'000'000, 20'000'000, 10'000'000,
                                    5'000'000, 1'000'000})
    .add_float64_axis("hit_ratio", {0.5, 1.0})
    .add_float64_axis("load_factor", {0.5});
