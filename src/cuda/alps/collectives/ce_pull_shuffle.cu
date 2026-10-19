// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/ce_pull_shuffle.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <mpi.h>

#include "alps/convert.hpp"
#include "alps/error.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "types/type_dispatcher.hpp"

namespace alps::collectives {
namespace {

using gamax::size_type;

constexpr std::size_t kArenaAlignment = 256;

std::size_t round_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0) return value;
    if (value > std::numeric_limits<std::size_t>::max() - alignment + 1) {
        throw std::overflow_error("CE-pull arena size overflow");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

CUdeviceptr add_bytes(CUdeviceptr pointer, std::size_t bytes) {
    return pointer + bytes;
}

void* as_pointer(CUdeviceptr pointer) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
}

const void* as_const_pointer(CUdeviceptr pointer) {
    return reinterpret_cast<const void*>(static_cast<std::uintptr_t>(pointer));
}

std::size_t ready_index(int world, int slot, int sender) {
    return static_cast<std::size_t>(slot) * 2 * world + sender;
}

std::size_t consumed_index(int world, int slot, int receiver) {
    return static_cast<std::size_t>(slot) * 2 * world + world + receiver;
}

CUdeviceptr control_word(CUdeviceptr base, std::size_t index) {
    return add_bytes(base, index * sizeof(unsigned long long));
}

__global__ void poll_geq_kernel(const unsigned long long* words, int count,
                                unsigned long long epoch) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    for (int i = lane; i < count; i += 32) {
        unsigned long long value = 0;
        do {
            value = __ldcg(words + i);
#if __CUDA_ARCH__ >= 700
            if (static_cast<long long>(value - epoch) < 0) __nanosleep(64);
#endif
        } while (static_cast<long long>(value - epoch) < 0);
    }
}

template <typename T>
__global__ void split_gather_kernel(
    const size_type* __restrict__ permutation,
    const T* __restrict__ input,
    size_type rows,
    const size_type* __restrict__ send_ends,
    int world,
    int rank,
    size_type self_send_begin,
    size_type self_send_count,
    size_type self_recv_begin,
    T* __restrict__ self_output,
    T* __restrict__ remote_output)
{
    size_type index = static_cast<size_type>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    const size_type stride =
        static_cast<size_type>(blockDim.x) * gridDim.x;
    for (; index < rows; index += stride) {
        int low = 0;
        int high = world;
        while (low < high) {
            const int middle = low + (high - low) / 2;
            if (index < send_ends[middle]) high = middle;
            else low = middle + 1;
        }
        const int peer = low;
        const T value = input[permutation[index]];
        if (peer == rank) {
            self_output[self_recv_begin + index - self_send_begin] = value;
        } else {
            const size_type compact_index =
                index - (peer > rank ? self_send_count : 0);
            remote_output[compact_index] = value;
        }
    }
}

__global__ void split_string_gather_kernel(
    const size_type* __restrict__ permutation,
    const size_type* __restrict__ input_offsets,
    const char* __restrict__ input_chars,
    const size_type* __restrict__ partitioned_offsets,
    size_type rows,
    const size_type* __restrict__ send_ends,
    int world,
    int rank,
    size_type self_send_begin,
    size_type self_send_count,
    size_type self_send_char_begin,
    size_type self_send_char_count,
    size_type self_recv_begin,
    size_type self_recv_char_begin,
    size_type* __restrict__ recv_lengths,
    char* __restrict__ recv_chars,
    size_type* __restrict__ remote_lengths,
    char* __restrict__ remote_chars)
{
    size_type index = static_cast<size_type>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    const size_type stride =
        static_cast<size_type>(blockDim.x) * gridDim.x;
    for (; index < rows; index += stride) {
        int low = 0;
        int high = world;
        while (low < high) {
            const int middle = low + (high - low) / 2;
            if (index < send_ends[middle]) high = middle;
            else low = middle + 1;
        }
        const int peer = low;
        const size_type source_row = permutation[index];
        const size_type source_begin = input_offsets[source_row];
        const size_type length =
            partitioned_offsets[index + 1] - partitioned_offsets[index];

        size_type row_destination = 0;
        size_type char_destination = 0;
        char* char_output = nullptr;
        if (peer == rank) {
            row_destination = self_recv_begin + index - self_send_begin;
            char_destination = self_recv_char_begin +
                partitioned_offsets[index] - self_send_char_begin;
            recv_lengths[row_destination] = length;
            char_output = recv_chars;
        } else {
            row_destination =
                index - (peer > rank ? self_send_count : 0);
            char_destination = partitioned_offsets[index] -
                (peer > rank ? self_send_char_count : 0);
            remote_lengths[row_destination] = length;
            char_output = remote_chars;
        }
        for (size_type byte = 0; byte < length; ++byte) {
            char_output[char_destination + byte] =
                input_chars[source_begin + byte];
        }
    }
}

__global__ void stage_allgather_string_lengths_kernel(
    const size_type* __restrict__ input_offsets,
    size_type rows,
    size_type self_recv_begin,
    size_type* __restrict__ recv_lengths,
    size_type* __restrict__ staged_lengths)
{
    size_type row = static_cast<size_type>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    const size_type stride =
        static_cast<size_type>(blockDim.x) * gridDim.x;
    for (; row < rows; row += stride) {
        const auto length = input_offsets[row + 1] - input_offsets[row];
        recv_lengths[self_recv_begin + row] = length;
        staged_lengths[row] = length;
    }
}

struct SplitGatherLauncher {
    template <typename T>
    void operator()(
        gamax::column_view permutation,
        gamax::column_view input,
        const size_type* send_ends,
        int world,
        int rank,
        size_type self_send_begin,
        size_type self_send_count,
        size_type self_recv_begin,
        gamax::mutable_column_view output,
        CUdeviceptr remote_output,
        cudaStream_t stream) const
    {
        const auto rows = input.size();
        if (rows == 0) return;
        constexpr int threads = 256;
        const auto needed = static_cast<unsigned long long>(
            (rows + threads - 1) / threads);
        const int blocks = static_cast<int>(std::min<unsigned long long>(
            needed, 65535));
        split_gather_kernel<T><<<blocks, threads, 0, stream>>>(
            permutation.data<size_type>(), input.data<T>(), rows, send_ends,
            world, rank, self_send_begin, self_send_count, self_recv_begin,
            output.data<T>(), reinterpret_cast<T*>(
                static_cast<std::uintptr_t>(remote_output)));
        ALPS_CUDACHECK(cudaGetLastError());
    }
};

class PinnedSizes {
public:
    explicit PinnedSizes(std::size_t count) : count_(count) {
        if (count_ != 0) {
            ALPS_CUDACHECK(cudaMallocHost(
                reinterpret_cast<void**>(&data_),
                count_ * sizeof(size_type)));
        }
    }
    ~PinnedSizes() { if (data_) cudaFreeHost(data_); }
    PinnedSizes(const PinnedSizes&) = delete;
    PinnedSizes& operator=(const PinnedSizes&) = delete;

    size_type* data() noexcept { return data_; }
    size_type& operator[](std::size_t index) noexcept { return data_[index]; }

private:
    size_type* data_ {nullptr};
    std::size_t count_ {0};
};

void preflight_ce_pull_shuffle(
    const HashPartitionPlan& plan,
    const std::vector<gamax::column_view>& columns,
    const Context& ctx,
    const CePullOptions& options)
{
    std::string local_error;
    auto fail = [&local_error](const char* message) {
        if (local_error.empty()) local_error = message;
    };

    try {
        if (ctx.size <= 0 || ctx.rank < 0 || ctx.rank >= ctx.size) {
            fail("invalid distributed context rank/size");
        }
        int current_device = -1;
        ALPS_CUDACHECK(cudaGetDevice(&current_device));
        if (current_device != ctx.local_device_id) {
            fail("active CUDA device differs from Context device");
        }
        if (options.slots < 1) fail("slot count must be positive");
        switch (options.schedule) {
            case CePullSchedule::Unordered:
            case CePullSchedule::RotatedDescriptors:
            case CePullSchedule::RoundRobinWaves:
                break;
            default:
                fail("invalid copy schedule");
        }
        if (!plan.permutation) {
            fail("partition plan has no permutation");
        }

        const auto world = ctx.size > 0
            ? static_cast<std::size_t>(ctx.size) : std::size_t{0};
        const auto& pattern = plan.pattern;
        if (pattern.send_counts.size() != world ||
            pattern.send_offsets.size() != world) {
            fail("send pattern does not match world size");
        }
        const bool recv_empty = pattern.recv_counts.empty() &&
            pattern.recv_offsets.empty();
        const bool recv_complete = pattern.recv_counts.size() == world &&
            pattern.recv_offsets.size() == world;
        if (!recv_empty && !recv_complete) {
            fail("receive pattern is partially populated");
        }

        if (plan.permutation &&
            pattern.send_counts.size() == world &&
            pattern.send_offsets.size() == world) {
            const auto rows = plan.permutation->size();
            size_type next = 0;
            for (std::size_t peer = 0; peer < world; ++peer) {
                const auto count = pattern.send_counts[peer];
                const auto offset = pattern.send_offsets[peer];
                if (count < 0 || offset != next || next > rows ||
                    count > rows - next) {
                    fail("send pattern is not a contiguous row partition");
                    break;
                }
                next += count;
            }
            if (next != rows) {
                fail("send pattern does not cover the permutation");
            }

            for (auto column : columns) {
                if (column.size() != rows) {
                    fail("column row count differs from partition plan");
                    break;
                }
                if (column.type().id() == gamax::type_id::STRING) {
                    if (column.auxiliary_tag() !=
                        static_cast<int32_t>(gamax::ARROW_STRING)) {
                        fail("STRING input does not use Arrow offsets");
                        break;
                    }
                } else if (gamax::size_of(column.type()) == 0) {
                    fail("unsupported zero-width column type");
                    break;
                }
            }
        }

        if (recv_complete) {
            size_type next = 0;
            for (std::size_t peer = 0; peer < world; ++peer) {
                const auto count = pattern.recv_counts[peer];
                const auto offset = pattern.recv_offsets[peer];
                if (count < 0 || offset != next ||
                    count > std::numeric_limits<size_type>::max() - next) {
                    fail("receive pattern is not a contiguous row layout");
                    break;
                }
                next += count;
            }
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown local preflight error";
    }

    const int local_ok = local_error.empty() ? 1 : 0;
    int global_ok = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_ok, &global_ok, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
    if (!global_ok) {
        if (!local_error.empty()) {
            std::fprintf(
                stderr, "[rank %d] CE-pull preflight: %s\n",
                ctx.rank, local_error.c_str());
        }
        throw std::invalid_argument(
            "begin_ce_pull_shuffle: rank-wide local preflight failed");
    }

    auto mix = [](std::uint64_t hash, std::uint64_t value) {
        constexpr std::uint64_t prime = 1099511628211ULL;
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (byte * 8)) & 0xffU;
            hash *= prime;
        }
        return hash;
    };
    std::uint64_t signature = 1469598103934665603ULL;
    signature = mix(signature, static_cast<std::uint64_t>(columns.size()));
    signature = mix(signature, static_cast<std::uint64_t>(options.slots));
    signature = mix(
        signature, static_cast<std::uint64_t>(options.doorbell_mode));
    signature = mix(
        signature, static_cast<std::uint64_t>(options.schedule));
    signature = mix(
        signature,
        static_cast<std::uint64_t>(options.high_priority_transfer_stream));
    signature = mix(
        signature,
        static_cast<std::uint64_t>(options.prefer_copy_compute_overlap));
    signature = mix(
        signature, static_cast<std::uint64_t>(options.full_staging));
    signature = mix(
        signature,
        static_cast<std::uint64_t>(!plan.pattern.recv_counts.empty()));
    for (auto column : columns) {
        signature = mix(
            signature, static_cast<std::uint64_t>(column.type().id()));
        signature = mix(
            signature,
            column.type().id() == gamax::type_id::STRING
                ? static_cast<std::uint64_t>(column.auxiliary_tag()) : 0);
    }
    std::uint64_t minimum_signature = 0;
    std::uint64_t maximum_signature = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &signature, &minimum_signature, 1, MPI_UINT64_T, MPI_MIN,
        ctx.mpi_comm));
    ALPS_MPICHECK(MPI_Allreduce(
        &signature, &maximum_signature, 1, MPI_UINT64_T, MPI_MAX,
        ctx.mpi_comm));
    if (minimum_signature != maximum_signature) {
        throw std::invalid_argument(
            "begin_ce_pull_shuffle: schema, logical column order, options, "
            "or receive-pattern state differs across ranks");
    }
}

void preflight_ce_pull_allgather(
    gamax::table_view input,
    const Context& ctx,
    const CePullOptions& options)
{
    std::string local_error;
    auto fail = [&local_error](const char* message) {
        if (local_error.empty()) local_error = message;
    };
    try {
        if (ctx.size <= 0 || ctx.rank < 0 || ctx.rank >= ctx.size) {
            fail("invalid distributed context rank/size");
        }
        int current_device = -1;
        ALPS_CUDACHECK(cudaGetDevice(&current_device));
        if (current_device != ctx.local_device_id) {
            fail("active CUDA device differs from Context device");
        }
        if (options.slots < 1) fail("slot count must be positive");
        if (options.full_staging) {
            fail("blocking allgather cannot use full staging");
        }
        switch (options.schedule) {
            case CePullSchedule::Unordered:
            case CePullSchedule::RotatedDescriptors:
            case CePullSchedule::RoundRobinWaves:
                break;
            default:
                fail("invalid copy schedule");
        }
        for (size_type column = 0; column < input.num_columns(); ++column) {
            const auto view = input.get_column(column);
            if (view.size() != input.num_rows()) {
                fail("column row count differs from table row count");
                break;
            }
            if (view.type().id() == gamax::type_id::STRING) {
                if (view.auxiliary_tag() !=
                    static_cast<int32_t>(gamax::ARROW_STRING)) {
                    fail("STRING input does not use Arrow offsets");
                    break;
                }
            } else if (gamax::size_of(view.type()) == 0) {
                fail("unsupported zero-width column type");
                break;
            }
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown local preflight error";
    }

    const int local_ok = local_error.empty() ? 1 : 0;
    int global_ok = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_ok, &global_ok, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
    if (!global_ok) {
        if (!local_error.empty()) {
            std::fprintf(
                stderr, "[rank %d] CE-pull allgather preflight: %s\n",
                ctx.rank, local_error.c_str());
        }
        throw std::invalid_argument(
            "ce_pull_allgather: rank-wide local preflight failed");
    }

    auto mix = [](std::uint64_t hash, std::uint64_t value) {
        constexpr std::uint64_t prime = 1099511628211ULL;
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (byte * 8)) & 0xffU;
            hash *= prime;
        }
        return hash;
    };
    std::uint64_t signature = 1469598103934665603ULL;
    signature = mix(
        signature, static_cast<std::uint64_t>(input.num_columns()));
    signature = mix(signature, static_cast<std::uint64_t>(options.slots));
    signature = mix(
        signature, static_cast<std::uint64_t>(options.doorbell_mode));
    signature = mix(
        signature, static_cast<std::uint64_t>(options.schedule));
    signature = mix(
        signature,
        static_cast<std::uint64_t>(options.high_priority_transfer_stream));
    signature = mix(
        signature,
        static_cast<std::uint64_t>(options.prefer_copy_compute_overlap));
    for (size_type column = 0; column < input.num_columns(); ++column) {
        const auto view = input.get_column(column);
        signature = mix(
            signature, static_cast<std::uint64_t>(view.type().id()));
        signature = mix(
            signature,
            view.type().id() == gamax::type_id::STRING
                ? static_cast<std::uint64_t>(view.auxiliary_tag()) : 0);
    }
    std::uint64_t minimum_signature = 0;
    std::uint64_t maximum_signature = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &signature, &minimum_signature, 1, MPI_UINT64_T, MPI_MIN,
        ctx.mpi_comm));
    ALPS_MPICHECK(MPI_Allreduce(
        &signature, &maximum_signature, 1, MPI_UINT64_T, MPI_MAX,
        ctx.mpi_comm));
    if (minimum_signature != maximum_signature) {
        throw std::invalid_argument(
            "ce_pull_allgather: schema, logical column order, or options "
            "differ across ranks");
    }
}

void coordinate_local_planning_error(
    std::exception_ptr local_error,
    const Context& ctx,
    const char* stage)
{
    const int local_ok = local_error ? 0 : 1;
    int global_ok = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_ok, &global_ok, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
    if (global_ok) return;

    if (local_error) {
        std::fprintf(
            stderr, "[rank %d] CE-pull local planning failed during %s\n",
            ctx.rank, stage);
        std::rethrow_exception(local_error);
    }
    throw std::runtime_error(
        std::string("begin_ce_pull_shuffle: another rank failed during ") +
        stage);
}

} // namespace

class CePullSessionImpl {
public:
    enum class Layout {
        Partitioned,
        AllGather,
    };

    struct ColumnState {
        gamax::column_view input;
        gamax::column_ptr output;
        gamax::column_ptr partitioned_string_offsets_owner;
        const size_type* partitioned_string_offsets {nullptr};
        gamax::column_ptr recv_string_lengths;
        gamax::device_buffer scan_scratch;
        std::size_t element_size {0};
        std::size_t slot_bytes {0};
        std::size_t remote_bytes {0};
        std::size_t string_char_slot_offset {0};
        std::size_t scan_scratch_bytes {0};
        size_type self_send_char_begin {0};
        size_type self_send_char_count {0};
        std::vector<size_type> recv_char_counts;
        std::vector<size_type> recv_char_offsets;
        std::vector<size_type> source_char_offsets;
        std::vector<std::size_t> source_char_slot_offsets;
        std::vector<void*> destinations;
        std::vector<const void*> sources;
        std::vector<std::size_t> sizes;
        cudaEvent_t ready {nullptr};
        std::size_t arena_offset {0};
        std::size_t lease_capacity {0};
        unsigned long long epoch {0};
        int slot {-1};
        bool is_string {false};
        bool prepared {false};
        bool pull_enqueued {false};

        ColumnState() = default;
        ColumnState(ColumnState&& other) noexcept
            : input(other.input), output(std::move(other.output)),
              partitioned_string_offsets_owner(
                  std::move(other.partitioned_string_offsets_owner)),
              partitioned_string_offsets(other.partitioned_string_offsets),
              recv_string_lengths(std::move(other.recv_string_lengths)),
              scan_scratch(std::move(other.scan_scratch)),
              element_size(other.element_size),
              slot_bytes(other.slot_bytes),
              remote_bytes(other.remote_bytes),
              string_char_slot_offset(other.string_char_slot_offset),
              scan_scratch_bytes(other.scan_scratch_bytes),
              self_send_char_begin(other.self_send_char_begin),
              self_send_char_count(other.self_send_char_count),
              recv_char_counts(std::move(other.recv_char_counts)),
              recv_char_offsets(std::move(other.recv_char_offsets)),
              source_char_offsets(std::move(other.source_char_offsets)),
              source_char_slot_offsets(
                  std::move(other.source_char_slot_offsets)),
              destinations(std::move(other.destinations)),
              sources(std::move(other.sources)), sizes(std::move(other.sizes)),
              ready(std::exchange(other.ready, nullptr)),
              arena_offset(other.arena_offset),
              lease_capacity(other.lease_capacity), epoch(other.epoch),
              slot(other.slot),
              is_string(other.is_string),
              prepared(other.prepared), pull_enqueued(other.pull_enqueued) {}
        ColumnState& operator=(ColumnState&&) = delete;
        ColumnState(const ColumnState&) = delete;
        ColumnState& operator=(const ColumnState&) = delete;
        ~ColumnState() { if (ready) cudaEventDestroy(ready); }
    };

    CePullSessionImpl(
        const HashPartitionPlan& plan,
        std::vector<gamax::column_view> columns,
        const Context& context,
        cudaStream_t preparation,
        const CePullOptions& options,
        bool input_is_partitioned = false)
        : ctx(context), preparation_stream(preparation),
          session_options(options), layout(Layout::Partitioned),
          permutation(plan.permutation ? plan.permutation->view()
                                       : gamax::column_view{}),
          pattern(plan.pattern)
    {
        if (!plan.permutation) {
            throw std::invalid_argument(
                "begin_ce_pull_shuffle: partition plan has no permutation");
        }
        complete_receive_pattern(pattern, ctx);
        const auto world = static_cast<std::size_t>(ctx.size);
        if (pattern.send_counts.size() != world ||
            pattern.send_offsets.size() != world ||
            pattern.recv_counts.size() != world ||
            pattern.recv_offsets.size() != world) {
            throw std::invalid_argument(
                "begin_ce_pull_shuffle: invalid communication pattern");
        }
        if (!ctx.ce_pull_context) {
            ctx.ce_pull_context = std::make_shared<CePullExchangeContext>();
        }
        exchange = ctx.ce_pull_context;
        exchange->initialize(ctx, options);

        std::vector<size_type> send_ends(world);
        std::vector<size_type> compact_offsets(world, 0);
        const auto self_count = pattern.send_counts[ctx.rank];
        for (int peer = 0; peer < ctx.size; ++peer) {
            send_ends[peer] =
                pattern.send_offsets[peer] + pattern.send_counts[peer];
            if (peer != ctx.rank) {
                compact_offsets[peer] = pattern.send_offsets[peer] -
                    (peer > ctx.rank ? self_count : 0);
            }
        }
        source_row_offsets.resize(world);
        ALPS_MPICHECK(MPI_Alltoall(
            compact_offsets.data(), 1, alps::convert::mpi_size_type(),
            source_row_offsets.data(), 1, alps::convert::mpi_size_type(),
            ctx.mpi_comm));

        const auto string_count = static_cast<std::size_t>(std::count_if(
            columns.begin(), columns.end(), [] (gamax::column_view column) {
                return column.type().id() == gamax::type_id::STRING;
            }));
        std::unique_ptr<PinnedSizes> string_boundaries;
        std::size_t required_slot_bytes = 0;
        std::size_t string_index = 0;
        const auto remote_peers = static_cast<std::size_t>(
            std::max(0, ctx.size - 1));
        std::size_t metadata_per_peer = 0;
        std::vector<size_type> send_metadata;
        std::vector<size_type> recv_metadata;
        std::exception_ptr local_planning_error;
        try {
            device_send_ends = gamax::device_buffer(
                world * sizeof(size_type), preparation_stream);
            ALPS_CUDACHECK(cudaMemcpyAsync(
                device_send_ends.data(), send_ends.data(),
                world * sizeof(size_type), cudaMemcpyHostToDevice,
                preparation_stream));
            string_boundaries = std::make_unique<PinnedSizes>(
                string_count * world * 2);
            states.reserve(columns.size());
            for (auto column : columns) {
                if (column.size() != permutation.size()) {
                    throw std::invalid_argument(
                        "begin_ce_pull_shuffle: column row count differs from plan");
                }
                ColumnState state;
                state.input = column;
                const auto remote_rows = column.size() - self_count;
                if (remote_rows < 0) {
                    throw std::overflow_error(
                        "begin_ce_pull_shuffle: negative remote row count");
                }
                state.is_string =
                    column.type().id() == gamax::type_id::STRING;
                if (state.is_string) {
                    if (column.auxiliary_tag() !=
                        static_cast<int32_t>(gamax::ARROW_STRING)) {
                        throw std::invalid_argument(
                            "begin_ce_pull_shuffle: STRING input must use Arrow offsets");
                    }
                    if (input_is_partitioned) {
                        state.partitioned_string_offsets =
                            static_cast<const size_type*>(
                                column.auxiliary_data());
                    } else {
                        auto gather_plan = gamax::plan_gather_strings(
                            gamax::string_column_view(
                                column, gamax::ARROW_STRING),
                            permutation, preparation_stream);
                        state.partitioned_string_offsets_owner =
                            std::move(gather_plan.offsets);
                        state.partitioned_string_offsets =
                            state.partitioned_string_offsets_owner
                                ->data<size_type>();
                        gather_plan.lengths->buffer().set_stream(
                            preparation_stream);
                        gather_plan.lengths.reset();
                    }

                    for (int peer = 0; peer < ctx.size; ++peer) {
                        const auto begin_row = pattern.send_offsets[peer];
                        const auto end_row =
                            begin_row + pattern.send_counts[peer];
                        const auto base =
                            (string_index * world +
                             static_cast<std::size_t>(peer)) * 2;
                        ALPS_CUDACHECK(cudaMemcpyAsync(
                            string_boundaries->data() + base,
                            state.partitioned_string_offsets +
                                begin_row,
                            sizeof(size_type), cudaMemcpyDeviceToHost,
                            preparation_stream));
                        ALPS_CUDACHECK(cudaMemcpyAsync(
                            string_boundaries->data() + base + 1,
                            state.partitioned_string_offsets +
                                end_row,
                            sizeof(size_type), cudaMemcpyDeviceToHost,
                            preparation_stream));
                    }
                    state.destinations.resize(remote_peers * 2);
                    state.sources.resize(remote_peers * 2);
                    state.sizes.resize(remote_peers * 2);
                    ++string_index;
                } else {
                    state.element_size = gamax::size_of(column.type());
                    if (state.element_size == 0 ||
                        static_cast<std::uint64_t>(remote_rows) >
                            std::numeric_limits<std::size_t>::max() /
                                state.element_size) {
                        throw std::overflow_error(
                            "begin_ce_pull_shuffle: fixed-width footprint overflow");
                    }
                    state.remote_bytes = static_cast<std::size_t>(remote_rows) *
                        state.element_size;
                    state.slot_bytes = state.remote_bytes;
                    state.output = gamax::make_column(
                        column.type(), pattern.total_recv(), preparation_stream);
                    state.destinations.resize(remote_peers);
                    state.sources.resize(remote_peers);
                    state.sizes.resize(remote_peers);
                    required_slot_bytes =
                        std::max(required_slot_bytes, state.slot_bytes);
                }
                ALPS_CUDACHECK(cudaEventCreateWithFlags(
                    &state.ready, cudaEventDisableTiming));
                states.push_back(std::move(state));
            }

            if (string_count != 0) {
                // One host rendezvous covers every string column's exact byte
                // boundaries; enqueue_column remains host-synchronization free.
                ALPS_CUDACHECK(cudaStreamSynchronize(preparation_stream));
                metadata_per_peer = string_count * 3;
                if (metadata_per_peer > static_cast<std::size_t>(
                        std::numeric_limits<int>::max())) {
                    throw std::overflow_error(
                        "begin_ce_pull_shuffle: too many STRING metadata fields");
                }
                send_metadata.assign(world * metadata_per_peer, 0);
                recv_metadata.assign(world * metadata_per_peer, 0);
                string_index = 0;
                for (auto& state : states) {
                    if (!state.is_string) continue;
                    const auto remote_rows = state.input.size() - self_count;
                    if (static_cast<std::uint64_t>(remote_rows) >
                        std::numeric_limits<std::size_t>::max() /
                            sizeof(size_type)) {
                        throw std::overflow_error(
                            "begin_ce_pull_shuffle: STRING length footprint overflow");
                    }
                    const auto length_bytes =
                        static_cast<std::size_t>(remote_rows) * sizeof(size_type);
                    state.string_char_slot_offset =
                        round_up(length_bytes, kArenaAlignment);
                    size_type self_chars = 0;
                    for (int peer = 0; peer < ctx.size; ++peer) {
                        const auto boundary_base =
                            (string_index * world +
                             static_cast<std::size_t>(peer)) * 2;
                        const auto begin = (*string_boundaries)[boundary_base];
                        const auto end =
                            (*string_boundaries)[boundary_base + 1];
                        if (begin < 0 || end < begin) {
                            throw std::runtime_error(
                                "begin_ce_pull_shuffle: invalid STRING boundary");
                        }
                        const auto count = end - begin;
                        if (peer == ctx.rank) {
                            state.self_send_char_begin = begin;
                            state.self_send_char_count = count;
                            self_chars = count;
                        }
                        const auto metadata_base =
                            static_cast<std::size_t>(peer) * metadata_per_peer +
                            string_index * 3;
                        send_metadata[metadata_base] = count;
                        send_metadata[metadata_base + 1] = begin;
                        send_metadata[metadata_base + 2] =
                            static_cast<size_type>(state.string_char_slot_offset);
                    }
                    for (int peer = ctx.rank + 1; peer < ctx.size; ++peer) {
                        const auto metadata_base =
                            static_cast<std::size_t>(peer) * metadata_per_peer +
                            string_index * 3;
                        send_metadata[metadata_base + 1] -= self_chars;
                    }

                    const auto total_chars = state.input.auxiliary_bytes();
                    if (total_chars < self_chars) {
                        throw std::runtime_error(
                            "begin_ce_pull_shuffle: STRING byte extent is invalid");
                    }
                    const auto remote_chars = total_chars - self_chars;
                    if (static_cast<std::uint64_t>(remote_chars) >
                        std::numeric_limits<std::size_t>::max() -
                            state.string_char_slot_offset) {
                        throw std::overflow_error(
                            "begin_ce_pull_shuffle: STRING character footprint overflow");
                    }
                    state.slot_bytes = state.string_char_slot_offset +
                        static_cast<std::size_t>(remote_chars);
                    state.remote_bytes = length_bytes +
                        static_cast<std::size_t>(remote_chars);
                    required_slot_bytes =
                        std::max(required_slot_bytes, state.slot_bytes);
                    ++string_index;
                }
            }

            const auto* injected = std::getenv(
                "ALPS_TEST_CE_PULL_FAIL_STAGE");
            if (ctx.rank == 0 && injected &&
                std::strcmp(injected, "before_string_metadata") == 0) {
                throw std::bad_alloc();
            }
        } catch (...) {
            local_planning_error = std::current_exception();
        }
        coordinate_local_planning_error(
            local_planning_error, ctx, "local column planning");

        if (string_count != 0) {
            ALPS_MPICHECK(MPI_Alltoall(
                send_metadata.data(), static_cast<int>(metadata_per_peer),
                alps::convert::mpi_size_type(),
                recv_metadata.data(), static_cast<int>(metadata_per_peer),
                alps::convert::mpi_size_type(), ctx.mpi_comm));

            std::exception_ptr local_output_error;
            try {
                string_index = 0;
                for (auto& state : states) {
                    if (!state.is_string) continue;
                    state.recv_char_counts.resize(world);
                    state.recv_char_offsets.resize(world);
                    state.source_char_offsets.resize(world);
                    state.source_char_slot_offsets.resize(world);
                    size_type recv_chars = 0;
                    for (int sender = 0; sender < ctx.size; ++sender) {
                        const auto metadata_base =
                            static_cast<std::size_t>(sender) * metadata_per_peer +
                            string_index * 3;
                        const auto count = recv_metadata[metadata_base];
                        const auto source_offset =
                            recv_metadata[metadata_base + 1];
                        const auto source_slot_offset =
                            recv_metadata[metadata_base + 2];
                        if (count < 0 || source_offset < 0 ||
                            source_slot_offset < 0 ||
                            recv_chars >
                                std::numeric_limits<size_type>::max() - count) {
                            throw std::overflow_error(
                                "begin_ce_pull_shuffle: invalid STRING receive metadata");
                        }
                        state.recv_char_counts[sender] = count;
                        state.recv_char_offsets[sender] = recv_chars;
                        state.source_char_offsets[sender] = source_offset;
                        state.source_char_slot_offsets[sender] =
                            static_cast<std::size_t>(source_slot_offset);
                        recv_chars += count;
                    }

                    gamax::device_buffer chars(
                        static_cast<std::size_t>(recv_chars),
                        preparation_stream);
                    const auto recv_rows = pattern.total_recv();
                    if (recv_rows < 0 || recv_rows ==
                            std::numeric_limits<size_type>::max() ||
                        static_cast<std::uint64_t>(recv_rows + 1) >
                            std::numeric_limits<std::size_t>::max() /
                                sizeof(size_type)) {
                        throw std::overflow_error(
                            "begin_ce_pull_shuffle: STRING offset footprint overflow");
                    }
                    gamax::device_buffer offsets(
                        static_cast<std::size_t>(recv_rows + 1) *
                            sizeof(size_type),
                        preparation_stream);
                    auto output = gamax::make_string_column(
                        std::move(chars), std::move(offsets), recv_rows,
                        preparation_stream, gamax::ARROW_STRING);
                    state.output = gamax::column_ptr(output.release());
                    state.recv_string_lengths = gamax::make_column<size_type>(
                        recv_rows, preparation_stream);
                    if (recv_rows != 0) {
                        auto* output_offsets = static_cast<size_type*>(
                            state.output->mutable_view().auxiliary_data());
                        ALPS_CUDACHECK(cub::DeviceScan::InclusiveSum(
                            nullptr, state.scan_scratch_bytes,
                            state.recv_string_lengths->data<size_type>(),
                            output_offsets + 1, recv_rows,
                            preparation_stream));
                        state.scan_scratch = gamax::device_buffer(
                            state.scan_scratch_bytes, preparation_stream);
                    }
                    ++string_index;
                }
            } catch (...) {
                local_output_error = std::current_exception();
            }
            coordinate_local_planning_error(
                local_output_error, ctx, "STRING receive allocation");
        }
        if (options.full_staging) {
            std::vector<std::size_t> local_bytes;
            local_bytes.reserve(states.size());
            for (const auto& state : states) {
                local_bytes.push_back(state.slot_bytes);
            }
            const auto leases = exchange->reserve_full_staging_leases(
                local_bytes, ctx, preparation_stream);
            if (leases.size() != states.size()) {
                throw std::logic_error(
                    "CE-pull full-staging lease count mismatch");
            }
            for (std::size_t index = 0; index < states.size(); ++index) {
                states[index].slot = leases[index].slot;
                states[index].arena_offset = leases[index].offset;
                states[index].lease_capacity = leases[index].capacity;
            }
        } else {
            exchange->ensure_slot_capacity(
                required_slot_bytes,
                static_cast<std::size_t>(options.slots), ctx);
        }
    }

    CePullSessionImpl(
        gamax::table_view input,
        const Context& context,
        cudaStream_t preparation,
        const CePullOptions& options)
        : ctx(context), preparation_stream(preparation),
          session_options(options), layout(Layout::AllGather)
    {
        const auto world = static_cast<std::size_t>(ctx.size);
        const auto local_rows = input.num_rows();
        pattern.send_counts.assign(world, local_rows);
        pattern.send_offsets.assign(world, 0);
        pattern.recv_counts.resize(world);
        ALPS_MPICHECK(MPI_Allgather(
            &local_rows, 1, alps::convert::mpi_size_type(),
            pattern.recv_counts.data(), 1, alps::convert::mpi_size_type(),
            ctx.mpi_comm));
        pattern.recv_offsets.resize(world);
        size_type recv_rows = 0;
        for (int sender = 0; sender < ctx.size; ++sender) {
            pattern.recv_offsets[sender] = recv_rows;
            if (pattern.recv_counts[sender] < 0 ||
                recv_rows > std::numeric_limits<size_type>::max() -
                    pattern.recv_counts[sender]) {
                throw std::overflow_error(
                    "ce_pull_allgather: receive row count overflow");
            }
            recv_rows += pattern.recv_counts[sender];
        }
        source_row_offsets.assign(world, 0);

        if (!ctx.ce_pull_context) {
            ctx.ce_pull_context = std::make_shared<CePullExchangeContext>();
        }
        exchange = ctx.ce_pull_context;
        exchange->initialize(ctx, options);

        const auto remote_peers = static_cast<std::size_t>(
            std::max(0, ctx.size - 1));
        std::size_t required_slot_bytes = 0;
        std::exception_ptr local_planning_error;
        try {
            states.reserve(static_cast<std::size_t>(input.num_columns()));
            for (size_type column = 0; column < input.num_columns(); ++column) {
                ColumnState state;
                state.input = input.get_column(column);
                state.is_string =
                    state.input.type().id() == gamax::type_id::STRING;
                if (state.is_string) {
                    if (local_rows < 0 ||
                        static_cast<std::uint64_t>(local_rows) >
                            std::numeric_limits<std::size_t>::max() /
                                sizeof(size_type)) {
                        throw std::overflow_error(
                            "ce_pull_allgather: STRING length footprint overflow");
                    }
                    const auto length_bytes = static_cast<std::size_t>(
                        local_rows) * sizeof(size_type);
                    state.string_char_slot_offset =
                        round_up(length_bytes, kArenaAlignment);
                    const auto local_chars = state.input.auxiliary_bytes();
                    if (local_chars < 0 ||
                        static_cast<std::uint64_t>(local_chars) >
                            std::numeric_limits<std::size_t>::max() -
                                state.string_char_slot_offset) {
                        throw std::overflow_error(
                            "ce_pull_allgather: STRING staging footprint overflow");
                    }
                    state.self_send_char_count = local_chars;
                    state.slot_bytes = state.string_char_slot_offset +
                        static_cast<std::size_t>(local_chars);
                    const auto transferred_bytes = length_bytes +
                        static_cast<std::size_t>(local_chars);
                    if (ctx.size > 1 && transferred_bytes >
                            std::numeric_limits<std::size_t>::max() /
                                static_cast<std::size_t>(ctx.size - 1)) {
                        throw std::overflow_error(
                            "ce_pull_allgather: STRING traffic footprint overflow");
                    }
                    state.remote_bytes = transferred_bytes *
                        static_cast<std::size_t>(std::max(0, ctx.size - 1));
                    state.recv_char_counts.resize(world);
                    state.recv_char_offsets.resize(world);
                    state.source_char_offsets.assign(world, 0);
                    state.source_char_slot_offsets.assign(
                        world, state.string_char_slot_offset);
                    state.destinations.resize(remote_peers * 2);
                    state.sources.resize(remote_peers * 2);
                    state.sizes.resize(remote_peers * 2);
                } else {
                    state.element_size = gamax::size_of(state.input.type());
                    if (state.element_size == 0 ||
                        static_cast<std::uint64_t>(local_rows) >
                            std::numeric_limits<std::size_t>::max() /
                                state.element_size) {
                        throw std::overflow_error(
                            "ce_pull_allgather: fixed-width staging footprint overflow");
                    }
                    state.slot_bytes = static_cast<std::size_t>(local_rows) *
                        state.element_size;
                    if (ctx.size > 1 && state.slot_bytes >
                            std::numeric_limits<std::size_t>::max() /
                                static_cast<std::size_t>(ctx.size - 1)) {
                        throw std::overflow_error(
                            "ce_pull_allgather: fixed-width traffic footprint overflow");
                    }
                    state.remote_bytes = state.slot_bytes *
                        static_cast<std::size_t>(std::max(0, ctx.size - 1));
                    state.output = gamax::make_column(
                        state.input.type(), recv_rows, preparation_stream);
                    state.destinations.resize(remote_peers);
                    state.sources.resize(remote_peers);
                    state.sizes.resize(remote_peers);
                }
                required_slot_bytes =
                    std::max(required_slot_bytes, state.slot_bytes);
                ALPS_CUDACHECK(cudaEventCreateWithFlags(
                    &state.ready, cudaEventDisableTiming));
                states.push_back(std::move(state));
            }
        } catch (...) {
            local_planning_error = std::current_exception();
        }
        coordinate_local_planning_error(
            local_planning_error, ctx, "allgather local column planning");

        for (auto& state : states) {
            if (!state.is_string) continue;
            const auto local_chars = state.self_send_char_count;
            ALPS_MPICHECK(MPI_Allgather(
                &local_chars, 1, alps::convert::mpi_size_type(),
                state.recv_char_counts.data(), 1,
                alps::convert::mpi_size_type(), ctx.mpi_comm));
            const auto local_char_slot_offset =
                static_cast<unsigned long long>(
                    state.string_char_slot_offset);
            std::vector<unsigned long long> source_char_slot_offsets(world);
            ALPS_MPICHECK(MPI_Allgather(
                &local_char_slot_offset, 1, MPI_UNSIGNED_LONG_LONG,
                source_char_slot_offsets.data(), 1,
                MPI_UNSIGNED_LONG_LONG, ctx.mpi_comm));
            for (std::size_t sender = 0; sender < world; ++sender) {
                if (source_char_slot_offsets[sender] >
                    std::numeric_limits<std::size_t>::max()) {
                    throw std::overflow_error(
                        "ce_pull_allgather: source STRING slot offset overflow");
                }
                state.source_char_slot_offsets[sender] =
                    static_cast<std::size_t>(
                        source_char_slot_offsets[sender]);
            }
        }

        std::exception_ptr local_output_error;
        try {
            for (auto& state : states) {
                if (!state.is_string) continue;
                size_type recv_chars = 0;
                for (int sender = 0; sender < ctx.size; ++sender) {
                    const auto count = state.recv_char_counts[sender];
                    if (count < 0 ||
                        recv_chars > std::numeric_limits<size_type>::max() -
                            count) {
                        throw std::overflow_error(
                            "ce_pull_allgather: STRING receive bytes overflow");
                    }
                    state.recv_char_offsets[sender] = recv_chars;
                    recv_chars += count;
                }
                if (recv_rows == std::numeric_limits<size_type>::max() ||
                    static_cast<std::uint64_t>(recv_rows + 1) >
                        std::numeric_limits<std::size_t>::max() /
                            sizeof(size_type)) {
                    throw std::overflow_error(
                        "ce_pull_allgather: STRING offset footprint overflow");
                }
                gamax::device_buffer chars(
                    static_cast<std::size_t>(recv_chars), preparation_stream);
                gamax::device_buffer offsets(
                    static_cast<std::size_t>(recv_rows + 1) *
                        sizeof(size_type),
                    preparation_stream);
                auto output = gamax::make_string_column(
                    std::move(chars), std::move(offsets), recv_rows,
                    preparation_stream, gamax::ARROW_STRING);
                state.output = gamax::column_ptr(output.release());
                state.recv_string_lengths = gamax::make_column<size_type>(
                    recv_rows, preparation_stream);
                if (recv_rows != 0) {
                    auto* output_offsets = static_cast<size_type*>(
                        state.output->mutable_view().auxiliary_data());
                    ALPS_CUDACHECK(cub::DeviceScan::InclusiveSum(
                        nullptr, state.scan_scratch_bytes,
                        state.recv_string_lengths->data<size_type>(),
                        output_offsets + 1, recv_rows, preparation_stream));
                    state.scan_scratch = gamax::device_buffer(
                        state.scan_scratch_bytes, preparation_stream);
                }
            }
        } catch (...) {
            local_output_error = std::current_exception();
        }
        coordinate_local_planning_error(
            local_output_error, ctx, "allgather receive allocation");
        exchange->ensure_slot_capacity(
            required_slot_bytes, static_cast<std::size_t>(options.slots), ctx);
    }

    void prepare(std::size_t ordinal) {
        if (ordinal >= states.size()) {
            throw std::out_of_range("CE-pull column ordinal out of range");
        }
        auto& state = states[ordinal];
        if (state.prepared) {
            throw std::logic_error("CE-pull column prepared twice");
        }

        const auto operation = exchange->next_operation_++;
        const int slot = session_options.full_staging
            ? state.slot
            : static_cast<int>(
                  operation %
                  static_cast<std::uint64_t>(session_options.slots));
        if (slot < 0 || static_cast<std::size_t>(slot) >=
                exchange->slots_.size()) {
            throw std::logic_error("CE-pull column has no valid arena lease");
        }
        const auto epoch = static_cast<unsigned long long>(operation + 1);
        auto& slot_state = exchange->slots_[slot];
        if (!session_options.full_staging &&
            operation >= static_cast<std::uint64_t>(session_options.slots)) {
            ++exchange->stats_.slot_reuse_count;
        }

        if (!session_options.full_staging) {
            // A previous full-staging batch may have used the same arena with
            // a different lease layout. Do not reuse bounded slots until its
            // whole-batch reclamation edge has completed.
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                preparation_stream, exchange->full_staging_batch_free_, 0));
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                preparation_stream, slot_state.free, 0));
            state.slot = slot;
            state.arena_offset = static_cast<std::size_t>(slot) *
                exchange->bounded_slot_capacity_bytes_;
            state.lease_capacity = exchange->bounded_slot_capacity_bytes_;
        }
        if (state.slot_bytes > state.lease_capacity) {
            throw std::logic_error(
                "CE-pull column footprint exceeds its arena lease");
        }
        state.epoch = epoch;
        const auto local_slot = exchange->arena_.allocated()
            ? add_bytes(exchange->arena_.own(), state.arena_offset)
            : CUdeviceptr{0};
        if (layout == Layout::AllGather) {
            if (state.is_string) {
                const auto rows = state.input.size();
                if (rows != 0) {
                    constexpr int threads = 256;
                    const auto needed = static_cast<unsigned long long>(
                        (rows + threads - 1) / threads);
                    const int blocks = static_cast<int>(
                        std::min<unsigned long long>(needed, 65535));
                    stage_allgather_string_lengths_kernel<<<
                        blocks, threads, 0, preparation_stream>>>(
                        static_cast<const size_type*>(
                            state.input.auxiliary_data()),
                        rows, pattern.recv_offsets[ctx.rank],
                        state.recv_string_lengths->data<size_type>(),
                        reinterpret_cast<size_type*>(
                            static_cast<std::uintptr_t>(local_slot)));
                    ALPS_CUDACHECK(cudaGetLastError());
                }
                const auto char_bytes = static_cast<std::size_t>(
                    state.self_send_char_count);
                if (char_bytes != 0) {
                    const auto staged_chars = add_bytes(
                        local_slot, state.string_char_slot_offset);
                    ALPS_CUDACHECK(cudaMemcpyAsync(
                        as_pointer(staged_chars), state.input.data<char>(),
                        char_bytes, cudaMemcpyDeviceToDevice,
                        preparation_stream));
                    ALPS_CUDACHECK(cudaMemcpyAsync(
                        state.output->data<char>() +
                            state.recv_char_offsets[ctx.rank],
                        state.input.data<char>(), char_bytes,
                        cudaMemcpyDeviceToDevice, preparation_stream));
                }
            } else {
                const auto bytes = state.slot_bytes;
                if (bytes != 0) {
                    ALPS_CUDACHECK(cudaMemcpyAsync(
                        as_pointer(local_slot), state.input.data(), bytes,
                        cudaMemcpyDeviceToDevice, preparation_stream));
                    ALPS_CUDACHECK(cudaMemcpyAsync(
                        static_cast<char*>(state.output->buffer().data()) +
                            static_cast<std::size_t>(
                                pattern.recv_offsets[ctx.rank]) *
                                state.element_size,
                        state.input.data(), bytes, cudaMemcpyDeviceToDevice,
                        preparation_stream));
                }
            }
        } else if (state.is_string) {
            const auto rows = state.input.size();
            if (rows != 0) {
                constexpr int threads = 256;
                const auto needed = static_cast<unsigned long long>(
                    (rows + threads - 1) / threads);
                const int blocks = static_cast<int>(
                    std::min<unsigned long long>(needed, 65535));
                const auto remote_chars = add_bytes(
                    local_slot, state.string_char_slot_offset);
                split_string_gather_kernel<<<
                    blocks, threads, 0, preparation_stream>>>(
                    permutation.data<size_type>(),
                    static_cast<const size_type*>(
                        state.input.auxiliary_data()),
                    state.input.data<char>(),
                    state.partitioned_string_offsets,
                    rows,
                    static_cast<const size_type*>(device_send_ends.data()),
                    ctx.size, ctx.rank,
                    pattern.send_offsets[ctx.rank],
                    pattern.send_counts[ctx.rank],
                    state.self_send_char_begin,
                    state.self_send_char_count,
                    pattern.recv_offsets[ctx.rank],
                    state.recv_char_offsets[ctx.rank],
                    state.recv_string_lengths->data<size_type>(),
                    state.output->data<char>(),
                    reinterpret_cast<size_type*>(
                        static_cast<std::uintptr_t>(local_slot)),
                    reinterpret_cast<char*>(
                        static_cast<std::uintptr_t>(remote_chars)));
                ALPS_CUDACHECK(cudaGetLastError());
            }
        } else {
            gamax::type_dispatcher(
                state.input.type(), SplitGatherLauncher{}, permutation,
                state.input,
                static_cast<const size_type*>(device_send_ends.data()),
                ctx.size, ctx.rank, pattern.send_offsets[ctx.rank],
                pattern.send_counts[ctx.rank], pattern.recv_offsets[ctx.rank],
                state.output->mutable_view(), local_slot, preparation_stream);
        }

        // Publish every READY before the local transfer stream starts waiting.
        for (int receiver = 0; receiver < ctx.size; ++receiver) {
            ALPS_CUCHECK(cuStreamWriteValue64(
                reinterpret_cast<CUstream>(preparation_stream),
                control_word(exchange->control_.peer(receiver),
                             ready_index(ctx.size, slot, ctx.rank)),
                epoch, CU_STREAM_WRITE_VALUE_DEFAULT));
        }
        ALPS_CUDACHECK(cudaEventRecord(
            slot_state.ready_published, preparation_stream));

        state.prepared = true;
    }

    void enqueue_pull(std::size_t ordinal) {
        if (ordinal >= states.size()) {
            throw std::out_of_range("CE-pull column ordinal out of range");
        }
        auto& state = states[ordinal];
        if (!state.prepared) {
            throw std::logic_error(
                "CE-pull column pull enqueued before preparation");
        }
        if (state.pull_enqueued) {
            throw std::logic_error("CE-pull column pull enqueued twice");
        }
        const auto slot = state.slot;
        const auto epoch = state.epoch;
        auto& slot_state = exchange->slots_[static_cast<std::size_t>(slot)];

        ALPS_CUDACHECK(cudaStreamWaitEvent(
            exchange->transfer_stream_, slot_state.ready_published, 0));
        exchange->wait_control_words(
            exchange->transfer_stream_, ready_index(ctx.size, slot, 0),
            ctx.size, epoch);

        std::size_t copy_count = 0;
        auto append_sender = [&] (int sender) {
            if (sender == ctx.rank) return;
            if (state.is_string) {
                if (pattern.recv_counts[sender] == 0 &&
                    state.recv_char_counts[sender] == 0) {
                    return;
                }
                const auto peer_slot = add_bytes(
                    exchange->arena_.peer(sender), state.arena_offset);
                if (pattern.recv_counts[sender] != 0) {
                    const auto bytes = static_cast<std::size_t>(
                        pattern.recv_counts[sender]) * sizeof(size_type);
                    state.destinations[copy_count] =
                        state.recv_string_lengths->data<size_type>() +
                        pattern.recv_offsets[sender];
                    state.sources[copy_count] = as_const_pointer(add_bytes(
                        peer_slot,
                        static_cast<std::size_t>(source_row_offsets[sender]) *
                            sizeof(size_type)));
                    state.sizes[copy_count] = bytes;
                    ++copy_count;
                }
                if (state.recv_char_counts[sender] != 0) {
                    state.destinations[copy_count] =
                        state.output->data<char>() +
                        state.recv_char_offsets[sender];
                    state.sources[copy_count] = as_const_pointer(add_bytes(
                        peer_slot,
                        state.source_char_slot_offsets[sender] +
                            static_cast<std::size_t>(
                                state.source_char_offsets[sender])));
                    state.sizes[copy_count] = static_cast<std::size_t>(
                        state.recv_char_counts[sender]);
                    ++copy_count;
                }
            } else if (pattern.recv_counts[sender] != 0) {
                const auto bytes = static_cast<std::size_t>(
                    pattern.recv_counts[sender]) * state.element_size;
                state.destinations[copy_count] =
                    static_cast<char*>(state.output->buffer().data()) +
                    static_cast<std::size_t>(pattern.recv_offsets[sender]) *
                        state.element_size;
                const auto peer_slot = add_bytes(
                    exchange->arena_.peer(sender), state.arena_offset);
                state.sources[copy_count] = as_const_pointer(add_bytes(
                    peer_slot,
                    static_cast<std::size_t>(source_row_offsets[sender]) *
                        state.element_size));
                state.sizes[copy_count] = bytes;
                ++copy_count;
            }
        };
        auto issue_batch = [&] (std::size_t first, std::size_t count) {
            if (count == 0) return;
            cudaMemcpyAttributes attributes{};
            attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            attributes.flags = session_options.prefer_copy_compute_overlap
                ? cudaMemcpyFlagPreferOverlapWithCompute
                : cudaMemcpyFlagDefault;
            std::size_t attributes_index = 0;
            ALPS_CUDACHECK(cudaMemcpyBatchAsync(
                state.destinations.data() + first,
                state.sources.data() + first,
                state.sizes.data() + first, count, &attributes,
                &attributes_index, 1, exchange->transfer_stream_));
        };

        if (session_options.schedule ==
            CePullSchedule::RoundRobinWaves) {
            // At wave p, receiver r pulls sender (r+p) mod world.  If ranks
            // remain approximately in step, every sender services one receiver
            // per wave instead of all receivers concentrating on sender zero.
            // Stream ordering serializes the waves without a host rendezvous.
            for (int phase = 1; phase < ctx.size; ++phase) {
                const auto first = copy_count;
                append_sender((ctx.rank + phase) % ctx.size);
                issue_batch(first, copy_count - first);
            }
        } else {
            if (session_options.schedule ==
                CePullSchedule::RotatedDescriptors) {
                for (int phase = 1; phase < ctx.size; ++phase) {
                    append_sender((ctx.rank + phase) % ctx.size);
                }
            } else {
                for (int sender = 0; sender < ctx.size; ++sender) {
                    append_sender(sender);
                }
            }
            issue_batch(0, copy_count);
        }
        if (!session_options.full_staging) {
            // The bounded-slot path can reuse this source slot as soon as
            // every receiver has finished this column's pull.
            for (int sender = 0; sender < ctx.size; ++sender) {
                ALPS_CUCHECK(cuStreamWriteValue64(
                    reinterpret_cast<CUstream>(exchange->transfer_stream_),
                    control_word(exchange->control_.peer(sender),
                                 consumed_index(ctx.size, slot, ctx.rank)),
                    epoch, CU_STREAM_WRITE_VALUE_DEFAULT));
            }
            ALPS_CUDACHECK(cudaEventRecord(
                slot_state.consumed_published, exchange->transfer_stream_));
        }

        if (state.is_string) {
            auto* output_offsets = static_cast<size_type*>(
                state.output->mutable_view().auxiliary_data());
            ALPS_CUDACHECK(cudaMemsetAsync(
                output_offsets, 0, sizeof(size_type),
                exchange->transfer_stream_));
            if (state.recv_string_lengths->size() != 0) {
                ALPS_CUDACHECK(cub::DeviceScan::InclusiveSum(
                    state.scan_scratch.data(), state.scan_scratch_bytes,
                    state.recv_string_lengths->data<size_type>(),
                    output_offsets + 1,
                    state.recv_string_lengths->size(),
                    exchange->transfer_stream_));
            }
            state.recv_string_lengths->buffer().set_stream(
                exchange->transfer_stream_);
            state.scan_scratch.set_stream(exchange->transfer_stream_);
            static_cast<gamax::string_column&>(*state.output)
                .meta_buffer().set_stream(exchange->transfer_stream_);
        }
        state.output->buffer().set_stream(exchange->transfer_stream_);
        ALPS_CUDACHECK(cudaEventRecord(
            state.ready, exchange->transfer_stream_));

        if (!session_options.full_staging) {
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                exchange->reclaim_stream_,
                slot_state.consumed_published, 0));
            exchange->wait_control_words(
                exchange->reclaim_stream_,
                consumed_index(ctx.size, slot, 0), ctx.size, epoch);
            ALPS_CUDACHECK(cudaEventRecord(
                slot_state.free, exchange->reclaim_stream_));
        }

        ++exchange->stats_.enqueued_columns;
        exchange->stats_.remote_bytes += state.remote_bytes;
        state.pull_enqueued = true;
    }

    void enqueue(std::size_t ordinal) {
        prepare(ordinal);
        enqueue_pull(ordinal);
    }

    const Context& ctx;
    cudaStream_t preparation_stream;
    CePullOptions session_options;
    Layout layout {Layout::Partitioned};
    gamax::column_view permutation;
    CommPattern pattern;
    std::shared_ptr<CePullExchangeContext> exchange;
    gamax::device_buffer device_send_ends;
    std::vector<size_type> source_row_offsets;
    std::vector<ColumnState> states;
};

CePullExchangeContext::~CePullExchangeContext() { teardown_local(); }

void CePullExchangeContext::initialize(
    const Context& ctx, const CePullOptions& options) {
    if (initialized_) {
        if (options.doorbell_mode != options_.doorbell_mode ||
            options.schedule != options_.schedule ||
            options.high_priority_transfer_stream !=
                options_.high_priority_transfer_stream ||
            options.prefer_copy_compute_overlap !=
                options_.prefer_copy_compute_overlap) {
            throw std::invalid_argument(
                "CE-pull stream/schedule options cannot change within one Context");
        }
        if (!options.full_staging) {
            if (bounded_slot_count_ != 0 &&
                bounded_slot_count_ != static_cast<std::size_t>(options.slots)) {
                throw std::invalid_argument(
                    "CE-pull bounded slot count cannot change within one Context");
            }
            bounded_slot_count_ = static_cast<std::size_t>(options.slots);
        }
        if (static_cast<std::size_t>(options.slots) > slots_.size()) {
            ensure_slot_count(static_cast<std::size_t>(options.slots), ctx);
        }
        return;
    }
    if (options.slots < 1) {
        throw std::invalid_argument("CE-pull slot count must be positive");
    }
    CUdevice device{};
    ALPS_CUCHECK(cuDeviceGet(&device, ctx.local_device_id));
    int fabric = 0;
    int stream64 = 0;
    ALPS_CUCHECK(cuDeviceGetAttribute(
        &fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    ALPS_CUCHECK(cuDeviceGetAttribute(
        &stream64, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device));
    // PollKernel replaces only the waiting side of the doorbell protocol.
    // Every mode publishes READY/completion with cuStreamWriteValue64.
    int local_ok = fabric && stream64;
    int global_ok = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_ok, &global_ok, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
    if (!global_ok) {
        throw std::runtime_error(
            "CE-pull requires fabric handles and 64-bit CUDA stream-memory "
            "operations; PollKernel replaces waits only");
    }

    options_ = options;
    if (!options.full_staging) {
        bounded_slot_count_ = static_cast<std::size_t>(options.slots);
    }
    world_ = ctx.size;
    rank_ = ctx.rank;

    int least_priority = 0;
    int greatest_priority = 0;
    ALPS_CUDACHECK(cudaDeviceGetStreamPriorityRange(
        &least_priority, &greatest_priority));
    const int priority = options.high_priority_transfer_stream
        ? greatest_priority : 0;
    ALPS_CUDACHECK(cudaStreamCreateWithPriority(
        &transfer_stream_, cudaStreamNonBlocking, priority));
    ALPS_CUDACHECK(cudaStreamCreateWithFlags(
        &reclaim_stream_, cudaStreamNonBlocking));
    ALPS_CUDACHECK(cudaEventCreateWithFlags(
        &full_staging_batch_free_, cudaEventDisableTiming));
    ALPS_CUDACHECK(cudaEventRecord(full_staging_batch_free_, ctx.stream));

    ensure_slot_count(static_cast<std::size_t>(options.slots), ctx);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    initialized_ = true;
}

void CePullExchangeContext::ensure_slot_count(
    std::size_t count, const Context& ctx) {
    if (count <= slots_.size()) return;
    if (stats_.growth_frozen) {
        ++stats_.arena_overflow_count;
        throw std::runtime_error(
            "CE-pull control-slot overflow after growth was frozen: required=" +
            std::to_string(count) + " capacity=" +
            std::to_string(slots_.size()));
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        count > std::numeric_limits<std::size_t>::max() /
            (2 * static_cast<std::size_t>(ctx.size) *
             sizeof(unsigned long long))) {
        throw std::overflow_error("CE-pull control-slot count overflow");
    }

    // Growing the fabric-mapped control buffer can change its mappings. Drain
    // every prior READY/CONSUMED user before the collective remap.
    if (transfer_stream_) {
        ALPS_CUDACHECK(cudaStreamSynchronize(transfer_stream_));
    }
    if (reclaim_stream_) {
        ALPS_CUDACHECK(cudaStreamSynchronize(reclaim_stream_));
    }

    const auto old_count = slots_.size();
    const auto old_words = old_count * 2 * static_cast<std::size_t>(ctx.size);
    const auto control_words =
        count * 2 * static_cast<std::size_t>(ctx.size);
    const auto old_capacity = control_.capacity_bytes();
    control_.ensure(control_words * sizeof(unsigned long long), ctx);
    stats_.control_capacity_bytes = control_.capacity_bytes();
    if (control_.capacity_bytes() != old_capacity) {
        ALPS_CUDACHECK(cudaMemsetAsync(
            as_pointer(control_.own()), 0,
            control_words * sizeof(unsigned long long), ctx.stream));
    } else {
        ALPS_CUDACHECK(cudaMemsetAsync(
            as_pointer(add_bytes(
                control_.own(), old_words * sizeof(unsigned long long))),
            0, (control_words - old_words) * sizeof(unsigned long long),
            ctx.stream));
    }

    slots_.resize(count);
    for (std::size_t index = old_count; index < count; ++index) {
        auto& slot = slots_[index];
        ALPS_CUDACHECK(cudaEventCreateWithFlags(
            &slot.ready_published, cudaEventDisableTiming));
        ALPS_CUDACHECK(cudaEventCreateWithFlags(
            &slot.free, cudaEventDisableTiming));
        ALPS_CUDACHECK(cudaEventCreateWithFlags(
            &slot.consumed_published, cudaEventDisableTiming));
        ALPS_CUDACHECK(cudaEventRecord(slot.free, ctx.stream));
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
    options_.slots = static_cast<int>(count);
}

void CePullExchangeContext::ensure_arena_capacity(
    std::size_t bytes, const Context& ctx) {
    if (bytes <= arena_.capacity_bytes()) {
        stats_.arena_capacity_bytes = arena_.capacity_bytes();
        return;
    }
    if (stats_.growth_frozen) {
        ++stats_.arena_overflow_count;
        throw std::runtime_error(
            "CE-pull arena overflow after growth was frozen: required=" +
            std::to_string(bytes) + " capacity=" +
            std::to_string(arena_.capacity_bytes()));
    }
    if (bytes == 0) return;
    const auto before_capacity = arena_.capacity_bytes();
    const auto begin = std::chrono::steady_clock::now();
    arena_.ensure(bytes, ctx);
    const auto end = std::chrono::steady_clock::now();
    if (arena_.capacity_bytes() != before_capacity) {
        stats_.arena_growth_ms +=
            std::chrono::duration<double, std::milli>(end - begin).count();
        ++stats_.arena_growth_count;
    }
    stats_.arena_capacity_bytes = arena_.capacity_bytes();
}

void CePullExchangeContext::wait_control_words(
    cudaStream_t stream, std::size_t first_index, int count,
    unsigned long long epoch) {
    if (options_.doorbell_mode == CeDoorbellMode::StreamMemOps) {
        for (int index = 0; index < count; ++index) {
            ALPS_CUCHECK(cuStreamWaitValue64(
                reinterpret_cast<CUstream>(stream),
                control_word(control_.own(), first_index + index),
                epoch, CU_STREAM_WAIT_VALUE_GEQ));
        }
    } else {
        const auto* words = reinterpret_cast<const unsigned long long*>(
            static_cast<std::uintptr_t>(control_word(
                control_.own(), first_index)));
        poll_geq_kernel<<<1, 32, 0, stream>>>(words, count, epoch);
        ALPS_CUDACHECK(cudaGetLastError());
    }
}

void CePullExchangeContext::begin_full_staging_batch(
    std::size_t operations, const Context& ctx,
    cudaStream_t preparation_stream, const CePullOptions& options) {
    if (!options.full_staging) {
        throw std::invalid_argument(
            "CE-pull full-staging batch requires full_staging=true");
    }
    initialize(ctx, options);
    if (full_staging_batch_active_) {
        throw std::logic_error("CE-pull full-staging batch is already active");
    }
    if (operations > slots_.size()) {
        ensure_slot_count(operations, ctx);
    }
    full_staging_expected_operations_ = operations;
    full_staging_reserved_operations_ = 0;
    full_staging_reserved_bytes_ = 0;
    full_staging_batch_active_ = true;
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        preparation_stream, full_staging_batch_free_, 0));
    // Bounded collectives reclaim each source slot independently. A following
    // full-staging batch may reuse the entire arena, so wait for every bounded
    // slot before assigning its alternate lease layout.
    for (std::size_t slot = 0;
         slot < std::min(slots_.size(),
                         bounded_slot_count_ == 0
                             ? slots_.size() : bounded_slot_count_);
         ++slot) {
        ALPS_CUDACHECK(cudaStreamWaitEvent(
            preparation_stream, slots_[slot].free, 0));
    }
}

std::vector<CePullExchangeContext::Lease>
CePullExchangeContext::reserve_full_staging_leases(
    const std::vector<std::size_t>& local_bytes, const Context& ctx,
    cudaStream_t preparation_stream) {
    if (!full_staging_batch_active_) {
        throw std::logic_error(
            "CE-pull full-staging lease requested outside an active batch");
    }
    if (local_bytes.size() >
        full_staging_expected_operations_ -
            full_staging_reserved_operations_) {
        throw std::logic_error(
            "CE-pull full-staging batch reserved too many columns");
    }

    std::vector<unsigned long long> local(local_bytes.begin(), local_bytes.end());
    std::vector<unsigned long long> global(local.size(), 0);
    if (!local.empty()) {
        ALPS_MPICHECK(MPI_Allreduce(
            local.data(), global.data(), static_cast<int>(local.size()),
            MPI_UNSIGNED_LONG_LONG, MPI_MAX, ctx.mpi_comm));
    }

    std::vector<Lease> leases;
    leases.reserve(global.size());
    for (const auto raw : global) {
        if (raw > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(
                "CE-pull full-staging lease footprint overflow");
        }
        const auto capacity = round_up(
            static_cast<std::size_t>(raw), kArenaAlignment);
        const auto offset = round_up(
            full_staging_reserved_bytes_, kArenaAlignment);
        if (capacity > std::numeric_limits<std::size_t>::max() - offset) {
            throw std::overflow_error(
                "CE-pull full-staging arena footprint overflow");
        }
        const auto slot_index = full_staging_reserved_operations_++;
        if (slot_index > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "CE-pull full-staging slot index overflow");
        }
        leases.push_back(Lease{
            static_cast<int>(slot_index), offset, capacity});
        full_staging_reserved_bytes_ = offset + capacity;
        stats_.peak_requested_slot_bytes = std::max(
            stats_.peak_requested_slot_bytes, capacity);
    }
    // Full-staging leases may replace the bounded arena layout. The begin
    // boundary placed all prior slot-free waits on preparation_stream; drain
    // them before a fabric mapping growth/remap. Avoid the host-side barrier
    // when the existing mapping is already large enough.
    if (full_staging_reserved_bytes_ > arena_.capacity_bytes()) {
        ALPS_CUDACHECK(cudaStreamSynchronize(preparation_stream));
    }
    ensure_arena_capacity(full_staging_reserved_bytes_, ctx);
    stats_.slot_capacity_bytes = std::max(
        stats_.slot_capacity_bytes, stats_.peak_requested_slot_bytes);
    return leases;
}

void CePullExchangeContext::seal_full_staging_batch() {
    if (!full_staging_batch_active_) {
        throw std::logic_error("CE-pull full-staging batch is not active");
    }
    if (full_staging_reserved_operations_ !=
        full_staging_expected_operations_) {
        throw std::logic_error(
            "CE-pull full-staging batch did not reserve every operation");
    }
}

void CePullExchangeContext::finish_full_staging_batch() {
    if (!full_staging_batch_active_) {
        throw std::logic_error("CE-pull full-staging batch is not active");
    }

    // Full staging does not reuse individual column leases. One completion
    // publication after all pulls is sufficient to protect the entire arena
    // from the next shuffle. The consumed words of control slot zero are
    // reserved for this whole-shuffle handshake in a full-staging context.
    const auto operation = next_operation_++;
    const auto epoch = static_cast<unsigned long long>(operation + 1);
    constexpr int completion_slot = 0;
    for (int sender = 0; sender < world_; ++sender) {
        ALPS_CUCHECK(cuStreamWriteValue64(
            reinterpret_cast<CUstream>(transfer_stream_),
            control_word(control_.peer(sender),
                         consumed_index(world_, completion_slot, rank_)),
            epoch, CU_STREAM_WRITE_VALUE_DEFAULT));
    }
    wait_control_words(
        reclaim_stream_, consumed_index(world_, completion_slot, 0),
        world_, epoch);
    ALPS_CUDACHECK(cudaEventRecord(
        full_staging_batch_free_, reclaim_stream_));
    full_staging_batch_active_ = false;
}

void CePullExchangeContext::cancel_full_staging_batch() noexcept {
    full_staging_expected_operations_ = 0;
    full_staging_reserved_operations_ = 0;
    full_staging_reserved_bytes_ = 0;
    full_staging_batch_active_ = false;
}

void CePullExchangeContext::ensure_slot_capacity(
    std::size_t bytes, std::size_t slot_count, const Context& ctx) {
    if (slot_count == 0 || slot_count > slots_.size()) {
        throw std::invalid_argument(
            "CE-pull bounded slot count is invalid");
    }
    if (bounded_slot_count_ != 0 && bounded_slot_count_ != slot_count) {
        throw std::invalid_argument(
            "CE-pull bounded slot count cannot change within one Context");
    }
    bounded_slot_count_ = slot_count;
    unsigned long long local = bytes;
    unsigned long long global = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ctx.mpi_comm));
    const auto required = round_up(
        static_cast<std::size_t>(global), kArenaAlignment);
    stats_.peak_requested_slot_bytes =
        std::max(stats_.peak_requested_slot_bytes, required);
    if (required <= bounded_slot_capacity_bytes_) return;
    if (stats_.growth_frozen) {
        ++stats_.arena_overflow_count;
        throw std::runtime_error(
            "CE-pull arena overflow after growth was frozen: required=" +
            std::to_string(required) + " capacity=" +
            std::to_string(bounded_slot_capacity_bytes_));
    }
    if (required == 0) return;
    if (required > std::numeric_limits<std::size_t>::max() / slot_count) {
        throw std::overflow_error("CE-pull total arena size overflow");
    }
    // A larger bounded collective can arrive while the previous one is still
    // draining asynchronously. Reclaim completion is transitively ordered
    // after source preparation and every receiver pull, so it is the safe host
    // boundary before changing the shared arena mapping.
    ALPS_CUDACHECK(cudaStreamSynchronize(reclaim_stream_));
    ALPS_CUDACHECK(cudaStreamSynchronize(transfer_stream_));
    ensure_arena_capacity(required * slot_count, ctx);
    // The VMM allocation is rounded as a whole.  Dividing that padded extent
    // directly by a non-power-of-two slot count can produce a misaligned slot
    // base (for example, three slots), which then faults typed partition and
    // join loads.  Round the stride down to our alignment; because the backing
    // is at least required*slots and required is aligned, this still fits every
    // requested slot and leaves only tail padding unused.
    const auto unaligned_stride =
        arena_.capacity_bytes() / slot_count;
    bounded_slot_capacity_bytes_ =
        (unaligned_stride / kArenaAlignment) * kArenaAlignment;
    if (bounded_slot_capacity_bytes_ < required) {
        throw std::runtime_error(
            "CE-pull arena padding produced an undersized slot stride");
    }
    stats_.slot_capacity_bytes = std::max(
        stats_.slot_capacity_bytes, bounded_slot_capacity_bytes_);
}

void CePullExchangeContext::freeze_growth() noexcept {
    stats_.growth_frozen = true;
}

void CePullExchangeContext::allow_growth() noexcept {
    stats_.growth_frozen = false;
}

CePullStats CePullExchangeContext::stats() const noexcept { return stats_; }

void CePullExchangeContext::teardown_local() noexcept {
    if (transfer_stream_) cudaStreamSynchronize(transfer_stream_);
    if (reclaim_stream_) cudaStreamSynchronize(reclaim_stream_);
    for (auto& slot : slots_) {
        if (slot.consumed_published) cudaEventDestroy(slot.consumed_published);
        if (slot.ready_published) cudaEventDestroy(slot.ready_published);
        if (slot.free) cudaEventDestroy(slot.free);
    }
    slots_.clear();
    if (full_staging_batch_free_) {
        cudaEventDestroy(full_staging_batch_free_);
    }
    if (reclaim_stream_) cudaStreamDestroy(reclaim_stream_);
    if (transfer_stream_) cudaStreamDestroy(transfer_stream_);
    reclaim_stream_ = nullptr;
    full_staging_batch_free_ = nullptr;
    full_staging_expected_operations_ = 0;
    full_staging_reserved_operations_ = 0;
    full_staging_reserved_bytes_ = 0;
    full_staging_batch_active_ = false;
    bounded_slot_capacity_bytes_ = 0;
    bounded_slot_count_ = 0;
    transfer_stream_ = nullptr;
    initialized_ = false;
}

void CePullExchangeContext::free(const Context& ctx) {
    if (!initialized_) return;
    ALPS_CUDACHECK(cudaStreamSynchronize(transfer_stream_));
    ALPS_CUDACHECK(cudaStreamSynchronize(reclaim_stream_));
    arena_.free(ctx);
    control_.free(ctx);
    teardown_local();
    stats_ = {};
    bounded_slot_capacity_bytes_ = 0;
    bounded_slot_count_ = 0;
    next_operation_ = 0;
}

CePullShuffleSession::CePullShuffleSession(
    std::unique_ptr<CePullSessionImpl> impl) : impl_(std::move(impl)) {}
CePullShuffleSession::~CePullShuffleSession() = default;
CePullShuffleSession::CePullShuffleSession(CePullShuffleSession&&) noexcept = default;
CePullShuffleSession& CePullShuffleSession::operator=(
    CePullShuffleSession&&) noexcept = default;

std::size_t CePullShuffleSession::num_columns() const noexcept {
    return impl_ ? impl_->states.size() : 0;
}

void CePullShuffleSession::prepare_column(std::size_t ordinal) {
    impl_->prepare(ordinal);
}

void CePullShuffleSession::enqueue_pull(std::size_t ordinal) {
    impl_->enqueue_pull(ordinal);
}

void CePullShuffleSession::enqueue_column(std::size_t ordinal) {
    impl_->enqueue(ordinal);
}

gamax::column& CePullShuffleSession::output(std::size_t ordinal) {
    if (ordinal >= impl_->states.size() || !impl_->states[ordinal].output) {
        throw std::out_of_range("CE-pull output ordinal is unavailable");
    }
    return *impl_->states[ordinal].output;
}

const gamax::column& CePullShuffleSession::output(std::size_t ordinal) const {
    if (ordinal >= impl_->states.size() || !impl_->states[ordinal].output) {
        throw std::out_of_range("CE-pull output ordinal is unavailable");
    }
    return *impl_->states[ordinal].output;
}

cudaEvent_t CePullShuffleSession::ready_event(std::size_t ordinal) const {
    if (ordinal >= impl_->states.size()) {
        throw std::out_of_range("CE-pull ready-event ordinal out of range");
    }
    return impl_->states[ordinal].ready;
}

void CePullShuffleSession::wait_on(
    std::size_t ordinal, cudaStream_t consumer_stream) const {
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        consumer_stream, ready_event(ordinal), 0));
}

void CePullShuffleSession::synchronize_reclamation() const {
    ALPS_CUDACHECK(cudaStreamSynchronize(impl_->exchange->transfer_stream_));
    ALPS_CUDACHECK(cudaStreamSynchronize(impl_->exchange->reclaim_stream_));
}

gamax::column_ptr CePullShuffleSession::release_output(
    std::size_t ordinal, cudaStream_t lifetime_stream) {
    if (ordinal >= impl_->states.size() || !impl_->states[ordinal].output) {
        throw std::out_of_range("CE-pull output ordinal is unavailable");
    }
    auto& output = *impl_->states[ordinal].output;
    output.buffer().set_stream(lifetime_stream);
    if (output.type().id() == gamax::type_id::STRING) {
        static_cast<gamax::string_column&>(output)
            .meta_buffer().set_stream(lifetime_stream);
    }
    return std::move(impl_->states[ordinal].output);
}

std::unique_ptr<CePullShuffleSession> begin_ce_pull_shuffle(
    const HashPartitionPlan& plan,
    std::vector<gamax::column_view> ordered_columns,
    const Context& ctx,
    cudaStream_t preparation_stream,
    const CePullOptions& options) {
    preflight_ce_pull_shuffle(plan, ordered_columns, ctx, options);
    auto impl = std::make_unique<CePullSessionImpl>(
        plan, std::move(ordered_columns), ctx, preparation_stream, options);
    return std::unique_ptr<CePullShuffleSession>(
        new CePullShuffleSession(std::move(impl)));
}

namespace {

std::unique_ptr<gamax::table> finish_blocking_session(
    std::unique_ptr<CePullShuffleSession> session,
    cudaStream_t stream,
    cudaEvent_t completion_event = nullptr)
{
    std::vector<gamax::column_ptr> columns(session->num_columns());
    for (std::size_t ordinal = 0; ordinal < session->num_columns(); ++ordinal) {
        session->enqueue_column(ordinal);
    }
    for (std::size_t ordinal = 0; ordinal < session->num_columns(); ++ordinal) {
        session->wait_on(ordinal, stream);
    }
    if (completion_event) {
        ALPS_CUDACHECK(cudaEventRecord(completion_event, stream));
    }
    // This adapter implements the Legacy execution contract: the collective
    // is complete before the operator returns, so neither downstream compute
    // nor planning for a following collective can race the bounded arena.
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    session->synchronize_reclamation();
    for (std::size_t ordinal = 0; ordinal < session->num_columns(); ++ordinal) {
        columns[ordinal] = session->release_output(ordinal, stream);
    }
    auto output = gamax::make_table(std::move(columns));
    return output;
}

} // namespace

std::unique_ptr<gamax::table> ce_pull_exchange_partitioned(
    gamax::table_view partitioned,
    CommPattern pattern,
    const Context& ctx,
    cudaStream_t stream,
    const CePullOptions& requested_options)
{
    CePullOptions options = requested_options;
    options.full_staging = false;
    complete_receive_pattern(pattern, ctx);

    HashPartitionPlan identity;
    identity.permutation = partitioned.num_rows() == 0
        ? gamax::make_column<size_type>(0, stream)
        : gamax::make_sequence<size_type>(
              partitioned.num_rows(), stream, size_type{0}, size_type{1});
    identity.pattern = std::move(pattern);
    std::vector<gamax::column_view> columns;
    columns.reserve(static_cast<std::size_t>(partitioned.num_columns()));
    for (size_type column = 0; column < partitioned.num_columns(); ++column) {
        columns.push_back(partitioned.get_column(column));
    }
    preflight_ce_pull_shuffle(identity, columns, ctx, options);
    auto impl = std::make_unique<CePullSessionImpl>(
        identity, std::move(columns), ctx, stream, options,
        /*input_is_partitioned=*/true);
    auto session = std::unique_ptr<CePullShuffleSession>(
        new CePullShuffleSession(std::move(impl)));
    return finish_blocking_session(std::move(session), stream);
}

std::unique_ptr<gamax::table> ce_pull_allgather(
    gamax::table_view input,
    const Context& ctx,
    cudaStream_t stream,
    const CePullOptions& requested_options,
    float* comm_ms)
{
    CePullOptions options = requested_options;
    options.full_staging = false;
    preflight_ce_pull_allgather(input, ctx, options);
    auto impl = std::make_unique<CePullSessionImpl>(
        input, ctx, stream, options);
    auto session = std::unique_ptr<CePullShuffleSession>(
        new CePullShuffleSession(std::move(impl)));
    cudaEvent_t begin_event = nullptr;
    cudaEvent_t end_event = nullptr;
    if (comm_ms) {
        ALPS_CUDACHECK(cudaEventCreate(&begin_event));
        ALPS_CUDACHECK(cudaEventCreate(&end_event));
        ALPS_CUDACHECK(cudaEventRecord(begin_event, stream));
    }
    auto result = finish_blocking_session(
        std::move(session), stream, end_event);
    if (comm_ms) {
        ALPS_CUDACHECK(cudaEventElapsedTime(
            comm_ms, begin_event, end_event));
        ALPS_CUDACHECK(cudaEventDestroy(begin_event));
        ALPS_CUDACHECK(cudaEventDestroy(end_event));
    }
    return result;
}

void begin_ce_pull_full_staging_batch(
    const Context& ctx,
    std::size_t operations,
    cudaStream_t preparation_stream,
    const CePullOptions& options) {
    if (!ctx.ce_pull_context) {
        ctx.ce_pull_context = std::make_shared<CePullExchangeContext>();
    }
    ctx.ce_pull_context->begin_full_staging_batch(
        operations, ctx, preparation_stream, options);
}

void seal_ce_pull_full_staging_batch(const Context& ctx) {
    if (!ctx.ce_pull_context) {
        throw std::logic_error("CE-pull full-staging context is unavailable");
    }
    ctx.ce_pull_context->seal_full_staging_batch();
}

void finish_ce_pull_full_staging_batch(const Context& ctx) {
    if (!ctx.ce_pull_context) {
        throw std::logic_error("CE-pull full-staging context is unavailable");
    }
    ctx.ce_pull_context->finish_full_staging_batch();
}

void cancel_ce_pull_full_staging_batch(const Context& ctx) noexcept {
    if (ctx.ce_pull_context) {
        ctx.ce_pull_context->cancel_full_staging_batch();
    }
}

void freeze_ce_pull_arena(const Context& ctx) {
    if (ctx.ce_pull_context) ctx.ce_pull_context->freeze_growth();
}

void allow_ce_pull_arena_growth(const Context& ctx) {
    if (ctx.ce_pull_context) ctx.ce_pull_context->allow_growth();
}

CePullStats ce_pull_stats(const Context& ctx) {
    return ctx.ce_pull_context ? ctx.ce_pull_context->stats() : CePullStats{};
}

cudaStream_t ce_pull_transfer_stream(const Context& ctx) {
    if (!ctx.ce_pull_context || !ctx.ce_pull_context->transfer_stream()) {
        throw std::logic_error("CE-pull transfer stream is not initialized");
    }
    return ctx.ce_pull_context->transfer_stream();
}

} // namespace alps::collectives
