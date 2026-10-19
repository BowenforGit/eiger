// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/operators/column_overlap_eq_join.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>

#include "alps/collectives/alltoall.hpp"
#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/collectives/table_exchange.hpp"
#include "alps/convert.hpp"
#include "alps/error.hpp"
#include "exec/step_profile.hpp"
#include "operators/join/inner_join.hpp"
#include "operators/join/materialize.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "utils/transform.hpp"

namespace alps::operators {
namespace {

using gamax::size_type;

struct Event {
    cudaEvent_t value {nullptr};

    explicit Event(bool create = true, bool timing = false) {
        if (create) {
            ALPS_CUDACHECK(cudaEventCreateWithFlags(
                &value, timing ? cudaEventDefault : cudaEventDisableTiming));
        }
    }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

class ProfilePhase {
public:
    ProfilePhase() = default;

    ProfilePhase(
        gamax::exec::StepProfileSink* sink,
        gamax::exec::PhaseMetadata metadata,
        cudaStream_t stream)
    {
        start(sink, std::move(metadata), stream);
    }

    ~ProfilePhase() noexcept {
        try {
            finish();
        } catch (...) {
            // Preserve an operator exception. Normal completion still reports
            // phase-contract failures through finish().
        }
    }

    ProfilePhase(const ProfilePhase&) = delete;
    ProfilePhase& operator=(const ProfilePhase&) = delete;

    void start(
        gamax::exec::StepProfileSink* sink,
        gamax::exec::PhaseMetadata metadata,
        cudaStream_t stream)
    {
        finish();
        sink_ = sink && sink->collects_phases() ? sink : nullptr;
        stream_ = stream;
        if (sink_) handle_ = sink_->begin_phase(std::move(metadata), stream_);
    }

    void finish() {
        if (sink_ &&
            handle_ != gamax::exec::StepProfileSink::invalid_phase_handle) {
            sink_->end_phase(handle_, stream_);
        }
        sink_ = nullptr;
        handle_ = gamax::exec::StepProfileSink::invalid_phase_handle;
    }

private:
    gamax::exec::StepProfileSink* sink_ {nullptr};
    gamax::exec::StepProfileSink::PhaseHandle handle_ {
        gamax::exec::StepProfileSink::invalid_phase_handle};
    cudaStream_t stream_ {cudaStreamDefault};
};

class NvtxRange {
public:
    explicit NvtxRange(const char* label) {
        nvtxRangePushA(label);
        active_ = true;
    }
    ~NvtxRange() { finish(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;

    void finish() noexcept {
        if (active_) nvtxRangePop();
        active_ = false;
    }

private:
    bool active_ {false};
};

void record_event(Event& event, cudaStream_t stream) {
    if (event.value) ALPS_CUDACHECK(cudaEventRecord(event.value, stream));
}

enum class Side { Left, Right };

struct ColumnRef {
    Side side;
    size_type ordinal;
};

const char* column_side_name(Side side) noexcept {
    return side == Side::Left ? "left" : "right";
}

void set_column_identity(
    gamax::exec::PhaseMetadata& metadata, ColumnRef ref)
{
    metadata.column_side = column_side_name(ref.side);
    metadata.column_ordinal = static_cast<std::int64_t>(ref.ordinal);
}

struct StringExchangeState {
    ColumnRef ref;
    gamax::column_ptr send_lengths;
    gamax::column_ptr recv_lengths;
    gamax::column_ptr recv_offsets;
    alps::collectives::CommPattern char_pattern;
};

struct StringOutputState {
    ColumnRef ref;
    gamax::string_gather_plan gather_plan;
};

class PinnedCounts {
public:
    explicit PinnedCounts(std::size_t count) : count_(count) {
        if (count_ > 0) {
            ALPS_CUDACHECK(cudaMallocHost(
                reinterpret_cast<void**>(&data_), count_ * sizeof(size_type)));
        }
    }
    ~PinnedCounts() { if (data_) cudaFreeHost(data_); }
    PinnedCounts(const PinnedCounts&) = delete;
    PinnedCounts& operator=(const PinnedCounts&) = delete;

    size_type* data() noexcept { return data_; }
    size_type& operator[](std::size_t index) noexcept { return data_[index]; }

private:
    size_type* data_ {nullptr};
    std::size_t count_ {0};
};

bool supported_hash_key(gamax::type_id id) {
    return id == gamax::type_id::INT32 || id == gamax::type_id::INT64 ||
           id == gamax::type_id::UINT32 || id == gamax::type_id::UINT64;
}

std::int64_t logical_column_bytes(gamax::column_view column) noexcept {
    if (column.type().id() == gamax::type_id::STRING) {
        return static_cast<std::int64_t>(column.auxiliary_bytes()) +
            static_cast<std::int64_t>(column.size() + 1) *
                static_cast<std::int64_t>(sizeof(gamax::size_type));
    }
    return static_cast<std::int64_t>(column.size()) *
        static_cast<std::int64_t>(gamax::size_of(column.type()));
}

void set_column_lifetime_stream(gamax::column& column, cudaStream_t stream) {
    column.buffer().set_stream(stream);
    if (column.type().id() == gamax::type_id::STRING) {
        static_cast<gamax::string_column&>(column).meta_buffer().set_stream(stream);
    }
}

std::vector<ColumnRef> payload_order(
    gamax::table_view left,
    size_type left_key,
    gamax::table_view right,
    size_type right_key,
    ColumnSchedule schedule)
{
    std::vector<ColumnRef> left_payloads;
    std::vector<ColumnRef> right_payloads;
    for (size_type column = 0; column < left.num_columns(); ++column) {
        if (column != left_key) left_payloads.push_back({Side::Left, column});
    }
    for (size_type column = 0; column < right.num_columns(); ++column) {
        if (column != right_key) right_payloads.push_back({Side::Right, column});
    }

    std::vector<ColumnRef> result;
    result.reserve(left_payloads.size() + right_payloads.size());
    if (schedule == ColumnSchedule::ProbeFirst) {
        result.insert(result.end(), right_payloads.begin(), right_payloads.end());
        result.insert(result.end(), left_payloads.begin(), left_payloads.end());
    } else if (schedule == ColumnSchedule::BuildFirst) {
        result.insert(result.end(), left_payloads.begin(), left_payloads.end());
        result.insert(result.end(), right_payloads.begin(), right_payloads.end());
    } else {
        const auto count = std::max(left_payloads.size(), right_payloads.size());
        for (std::size_t index = 0; index < count; ++index) {
            if (index < right_payloads.size()) result.push_back(right_payloads[index]);
            if (index < left_payloads.size()) result.push_back(left_payloads[index]);
        }
    }
    return result;
}

void validate_local(
    gamax::table_view left,
    size_type left_key,
    gamax::table_view right,
    size_type right_key,
    const gamax::join::join_hint& hint)
{
    if (left.num_columns() == 0 || right.num_columns() == 0) {
        throw std::invalid_argument(
            "column_overlap_eq_join: inputs must have columns");
    }
    if (left_key < 0 || left_key >= left.num_columns() ||
        right_key < 0 || right_key >= right.num_columns()) {
        throw std::out_of_range(
            "column_overlap_eq_join: key ordinal out of range");
    }
    if (hint.join_algo != gamax::join::HASH) {
        throw std::invalid_argument(
            "column_overlap_eq_join: only HASH is supported");
    }
    const auto left_type = left.get_column(left_key).type();
    const auto right_type = right.get_column(right_key).type();
    if (left_type != right_type || !supported_hash_key(left_type.id())) {
        throw std::invalid_argument(
            "column_overlap_eq_join: unsupported or mismatched key types");
    }
    auto validate_payloads = [] (gamax::table_view input) {
        for (size_type column = 0; column < input.num_columns(); ++column) {
            const auto view = input.get_column(column);
            if (view.type().id() == gamax::type_id::STRING &&
                view.auxiliary_tag() !=
                    static_cast<int32_t>(gamax::ARROW_STRING)) {
                throw std::invalid_argument(
                    "column_overlap_eq_join: STRING payloads must use Arrow offsets");
            }
        }
    };
    validate_payloads(left);
    validate_payloads(right);
}

bool require_rank_wide_success(
    bool local_success,
    const alps::Context& context,
    const char* operation)
{
    int local = local_success ? 1 : 0;
    int global = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local, &global, 1, MPI_INT, MPI_MIN, context.mpi_comm));
    if (!global && local_success) {
        throw std::runtime_error(
            std::string("column_overlap_eq_join: rank-wide failure during ") +
            operation);
    }
    return global != 0;
}

bool rank_wide_stage_succeeded(
    bool local_success, const alps::Context& context)
{
    int local = local_success ? 1 : 0;
    int global = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local, &global, 1, MPI_INT, MPI_MIN, context.mpi_comm));
    return global != 0;
}

void drain_runtime(ColumnJoinRuntime& runtime) noexcept {
    // Every coordinated boundary is placed after all ranks have enqueued the
    // same NCCL sequence. Drain those operations before unwinding so a caught
    // allocation failure cannot poison the reusable runtime or communicator.
    cudaStreamSynchronize(runtime.communication_stream);
    cudaStreamSynchronize(runtime.preparation_stream);
    cudaStreamSynchronize(runtime.join_stream);
}

[[noreturn]] void throw_rank_wide_stage_failure(
    std::exception_ptr local_error, const char* operation)
{
    if (local_error) std::rethrow_exception(local_error);
    throw std::runtime_error(
        std::string("column_overlap_eq_join: another rank failed during ") +
        operation);
}

void inject_test_failure(const alps::Context& context, const char* stage) {
    const auto* requested =
        std::getenv("ALPS_TEST_COLUMN_OVERLAP_FAIL_STAGE");
    if (context.rank == 0 && requested && std::strcmp(requested, stage) == 0) {
        throw std::bad_alloc();
    }
}

float elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float value = 0.0f;
    ALPS_CUDACHECK(cudaEventElapsedTime(&value, start, stop));
    return value;
}

} // namespace

ColumnJoinRuntime::ColumnJoinRuntime(bool high_priority)
    : communication_high_priority(high_priority)
{
    if (high_priority) {
        int least_priority = 0;
        int greatest_priority = 0;
        ALPS_CUDACHECK(cudaDeviceGetStreamPriorityRange(
            &least_priority, &greatest_priority));
        (void)least_priority;
        ALPS_CUDACHECK(cudaStreamCreateWithPriority(
            &communication_stream, cudaStreamNonBlocking, greatest_priority));
    } else {
        ALPS_CUDACHECK(cudaStreamCreateWithFlags(
            &communication_stream, cudaStreamNonBlocking));
    }
    try {
        ALPS_CUDACHECK(cudaStreamCreateWithFlags(
            &preparation_stream, cudaStreamNonBlocking));
        ALPS_CUDACHECK(cudaStreamCreateWithFlags(
            &join_stream, cudaStreamNonBlocking));
        nvtxNameCudaStreamA(communication_stream, "alps.column_join.comm");
        nvtxNameCudaStreamA(preparation_stream, "alps.column_join.prepare");
        nvtxNameCudaStreamA(join_stream, "alps.column_join.match_materialize");
    } catch (...) {
        if (join_stream) cudaStreamDestroy(join_stream);
        if (preparation_stream) cudaStreamDestroy(preparation_stream);
        if (communication_stream) cudaStreamDestroy(communication_stream);
        throw;
    }
}

ColumnJoinRuntime::~ColumnJoinRuntime() {
    if (communication_stream) cudaStreamSynchronize(communication_stream);
    if (preparation_stream) cudaStreamSynchronize(preparation_stream);
    if (join_stream) cudaStreamSynchronize(join_stream);
    if (join_stream) cudaStreamDestroy(join_stream);
    if (preparation_stream) cudaStreamDestroy(preparation_stream);
    if (communication_stream) cudaStreamDestroy(communication_stream);
    for (auto event : dependency_events) cudaEventDestroy(event);
}

void ColumnJoinRuntime::ensure_dependency_events(std::size_t count) {
    while (dependency_events.size() < count) {
        cudaEvent_t event = nullptr;
        ALPS_CUDACHECK(cudaEventCreateWithFlags(
            &event, cudaEventDisableTiming));
        dependency_events.push_back(event);
    }
}

namespace {

std::vector<gamax::column_view> column_views(gamax::table_view table) {
    std::vector<gamax::column_view> result;
    result.reserve(static_cast<std::size_t>(table.num_columns()));
    for (size_type ordinal = 0; ordinal < table.num_columns(); ++ordinal) {
        result.push_back(table.get_column(ordinal));
    }
    return result;
}

std::unique_ptr<gamax::table> column_overlap_eq_join_ce_pull(
    gamax::table_view left,
    size_type left_key,
    gamax::table_view right,
    size_type right_key,
    const gamax::join::join_hint& hint,
    const alps::Context& context,
    cudaStream_t caller_stream,
    bool partition_left,
    bool partition_right,
    ColumnSchedule schedule,
    ColumnOverlapTiming* timing,
    gamax::exec::StepProfileSink* profile_sink,
    bool communication_high_priority,
    bool early_reclaim,
    alps::collectives::CePullSchedule ce_pull_schedule,
    int ce_pull_slots)
{
    Event total_start(timing != nullptr, true);
    Event total_stop(timing != nullptr, true);
    Event key_prepare_start(timing != nullptr, true);
    Event key_prepare_stop(timing != nullptr, true);
    Event payload_prepare_start(timing != nullptr, true);
    Event payload_prepare_stop(timing != nullptr, true);
    Event key_exchange_start(timing != nullptr, true);
    Event key_exchange_stop(timing != nullptr, true);
    Event payload_exchange_start(timing != nullptr, true);
    Event payload_exchange_stop(timing != nullptr, true);
    Event match_start(timing != nullptr, true);
    Event match_stop(timing != nullptr, true);
    Event materialize_start(timing != nullptr, true);
    Event materialize_stop(timing != nullptr, true);

    record_event(total_start, caller_stream);
    alps::collectives::HashPartitionPlan left_plan;
    alps::collectives::HashPartitionPlan right_plan;
    std::exception_ptr local_error;
    ProfilePhase partition_phase(
        profile_sink,
        {"join.column.partition_plan", gamax::exec::PhaseCategory::Compute},
        caller_stream);
    try {
        validate_local(left, left_key, right, right_key, hint);
        if (!context.column_join_runtime) {
            context.column_join_runtime = std::make_shared<ColumnJoinRuntime>(
                communication_high_priority);
        } else if (context.column_join_runtime->communication_high_priority !=
                   communication_high_priority) {
            throw std::invalid_argument(
                "column_overlap_eq_join: communication priority cannot change "
                "within one Context");
        }
        context.column_join_runtime->ensure_dependency_events(4);
        if (partition_left) {
            left_plan = alps::collectives::make_hash_partition_plan(
                left.select_columns({left_key}), context, caller_stream);
        }
        if (partition_right) {
            right_plan = alps::collectives::make_hash_partition_plan(
                right.select_columns({right_key}), context, caller_stream);
        }
    } catch (...) {
        local_error = std::current_exception();
    }
    partition_phase.finish();
    require_rank_wide_success(
        !local_error, context, "CE-pull validation and partition planning");
    if (local_error) std::rethrow_exception(local_error);

    auto& runtime = *context.column_join_runtime;
    const auto entry = runtime.dependency_events[0];
    const auto final_event = runtime.dependency_events[1];
    const auto all_payloads_prepared = runtime.dependency_events[2];
    const auto retrieve_start = runtime.dependency_events[3];
    ALPS_CUDACHECK(cudaEventRecord(entry, caller_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.preparation_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(runtime.join_stream, entry, 0));

    alps::collectives::CePullOptions ce_options;
    const auto full_staging_operations =
        (partition_left ? static_cast<std::size_t>(left.num_columns()) : 0) +
        (partition_right ? static_cast<std::size_t>(right.num_columns()) : 0);
    if (full_staging_operations > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "column_overlap_eq_join: too many full-staging operations");
    }
    ce_options.slots = std::max(
        ce_pull_slots, static_cast<int>(full_staging_operations));
    ce_options.high_priority_transfer_stream = communication_high_priority;
    ce_options.schedule = ce_pull_schedule;
    ce_options.full_staging = true;
    std::unique_ptr<alps::collectives::CePullShuffleSession> left_session;
    std::unique_ptr<alps::collectives::CePullShuffleSession> right_session;
    local_error = nullptr;
    try {
        // This is the sole host-planning boundary for the transport. Both
        // sessions allocate every output and exchange all string metadata
        // before the first stream-ordered column operation is issued.
        alps::collectives::begin_ce_pull_full_staging_batch(
            context, full_staging_operations, runtime.preparation_stream,
            ce_options);
        if (partition_left) {
            left_session = alps::collectives::begin_ce_pull_shuffle(
                left_plan, column_views(left), context,
                runtime.preparation_stream, ce_options);
        }
        if (partition_right) {
            right_session = alps::collectives::begin_ce_pull_shuffle(
                right_plan, column_views(right), context,
                runtime.preparation_stream, ce_options);
        }
        alps::collectives::seal_ce_pull_full_staging_batch(context);
    } catch (...) {
        local_error = std::current_exception();
    }
    if (!rank_wide_stage_succeeded(!local_error, context)) {
        alps::collectives::cancel_ce_pull_full_staging_batch(context);
        throw_rank_wide_stage_failure(
            local_error, "CE-pull full-staging session planning");
    }

    const auto transfer_stream = (left_session || right_session)
        ? alps::collectives::ce_pull_transfer_stream(context)
        : runtime.communication_stream;
    if (left_session || right_session) {
        ALPS_CUDACHECK(cudaStreamWaitEvent(transfer_stream, entry, 0));
    }

    std::vector<ColumnRef> issue_order;
    if (partition_left) issue_order.push_back({Side::Left, left_key});
    if (partition_right) issue_order.push_back({Side::Right, right_key});
    const auto key_issue_count = issue_order.size();
    auto payloads = payload_order(left, left_key, right, right_key, schedule);
    for (const auto ref : payloads) {
        if ((ref.side == Side::Left && partition_left) ||
            (ref.side == Side::Right && partition_right)) {
            issue_order.push_back(ref);
        }
    }

    // Full staging deliberately separates the SM-backed split/gather work
    // from the copy-engine pulls.  Every payload partition is submitted before
    // the hash count starts; only key pulls are released immediately.
    record_event(key_prepare_start, runtime.preparation_stream);
    record_event(key_exchange_start, transfer_stream);
    bool payload_prepare_started = false;
    if (key_issue_count == 0) {
        record_event(key_prepare_stop, runtime.preparation_stream);
        record_event(key_exchange_stop, transfer_stream);
        record_event(payload_prepare_start, runtime.preparation_stream);
        payload_prepare_started = true;
    }
    NvtxRange prepare_range("column_overlap.ce_pull.prepare_all");
    for (std::size_t index = 0; index < issue_order.size(); ++index) {
        if (index == key_issue_count) {
            record_event(key_prepare_stop, runtime.preparation_stream);
            record_event(key_exchange_stop, transfer_stream);
            record_event(payload_prepare_start, runtime.preparation_stream);
            payload_prepare_started = true;
        }
        const auto ref = issue_order[index];
        auto& session = ref.side == Side::Left ? left_session : right_session;
        const auto input = ref.side == Side::Left
            ? left.get_column(ref.ordinal) : right.get_column(ref.ordinal);

        gamax::exec::PhaseMetadata prepare_metadata{
            "join.column.ce_pull_partition",
            gamax::exec::PhaseCategory::Compute};
        set_column_identity(prepare_metadata, ref);
        ProfilePhase prepare_phase(
            profile_sink, std::move(prepare_metadata),
            runtime.preparation_stream);
        session->prepare_column(static_cast<std::size_t>(ref.ordinal));
        prepare_phase.finish();

        if (index < key_issue_count) {
            const auto& output = session->output(
                static_cast<std::size_t>(ref.ordinal));
            gamax::exec::PhaseMetadata exchange_metadata{
                "join.column.ce_pull_exchange",
                gamax::exec::PhaseCategory::Communication,
                gamax::exec::CommunicationPattern::Shuffle,
                "cuda_memcpy_batch_pull",
                logical_column_bytes(input),
                logical_column_bytes(output.view()),
                input.size(), output.size(), context.size - 1};
            set_column_identity(exchange_metadata, ref);
            ProfilePhase exchange_phase(
                profile_sink, std::move(exchange_metadata), transfer_stream);
            session->enqueue_pull(static_cast<std::size_t>(ref.ordinal));
            exchange_phase.finish();
        }
    }
    if (!payload_prepare_started) {
        record_event(key_prepare_stop, runtime.preparation_stream);
        record_event(key_exchange_stop, transfer_stream);
        record_event(payload_prepare_start, runtime.preparation_stream);
    }
    record_event(payload_prepare_stop, runtime.preparation_stream);
    ALPS_CUDACHECK(cudaEventRecord(
        all_payloads_prepared, runtime.preparation_stream));
    prepare_range.finish();

    std::vector<gamax::column_ptr> left_recv(
        static_cast<std::size_t>(left.num_columns()));
    std::vector<gamax::column_ptr> right_recv(
        static_cast<std::size_t>(right.num_columns()));
    auto release_received = [&] (ColumnRef ref) {
        auto& session = ref.side == Side::Left ? left_session : right_session;
        if (!session) return;
        auto& received = ref.side == Side::Left
            ? left_recv[static_cast<std::size_t>(ref.ordinal)]
            : right_recv[static_cast<std::size_t>(ref.ordinal)];
        if (!received) {
            received = session->release_output(
                static_cast<std::size_t>(ref.ordinal), runtime.join_stream);
        }
    };

    // The match consumes only the shuffled keys.  Payload ownership remains
    // in the sessions until their delayed pulls have been submitted.
    release_received({Side::Left, left_key});
    release_received({Side::Right, right_key});

    auto input_for = [&] (ColumnRef ref) -> gamax::column_view {
        if (ref.side == Side::Left) {
            if (!partition_left) return left.get_column(ref.ordinal);
            const auto& received =
                left_recv[static_cast<std::size_t>(ref.ordinal)];
            if (!received) {
                throw std::logic_error(
                    "column_overlap_eq_join: left CE output not released");
            }
            return received->view();
        }
        if (!partition_right) return right.get_column(ref.ordinal);
        const auto& received =
            right_recv[static_cast<std::size_t>(ref.ordinal)];
        if (!received) {
            throw std::logic_error(
                "column_overlap_eq_join: right CE output not released");
        }
        return received->view();
    };
    auto wait_for = [&] (ColumnRef ref) {
        if (ref.side == Side::Left && left_session) {
            left_session->wait_on(
                static_cast<std::size_t>(ref.ordinal), runtime.join_stream);
        } else if (ref.side == Side::Right && right_session) {
            right_session->wait_on(
                static_cast<std::size_t>(ref.ordinal), runtime.join_stream);
        }
    };

    wait_for({Side::Left, left_key});
    wait_for({Side::Right, right_key});
    // Count is intentionally held until all payload split/gathers are done so
    // they cannot be starved by the count/retrieve kernels.  The prepared VMM
    // leases remain live while count executes.
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.join_stream, all_payloads_prepared, 0));

    bool payload_pulls_enqueued = false;
    gamax::join::BeforeProbeHook release_payload_pulls =
        [&] (cudaStream_t retrieve_stream) {
            if (payload_pulls_enqueued) {
                throw std::logic_error(
                    "column_overlap_eq_join: payload pulls released twice");
            }

            // This event is ordered after count and immediately before the
            // retrieve launch.  It turns the otherwise one-sided memcpy calls
            // into the stream-ordered release point for all payload pulls.
            ALPS_CUDACHECK(cudaEventRecord(retrieve_start, retrieve_stream));
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                transfer_stream, retrieve_start, 0));
            record_event(payload_exchange_start, transfer_stream);
            NvtxRange exchange_range(
                "column_overlap.ce_pull.release_payload_pulls");
            for (std::size_t index = key_issue_count;
                 index < issue_order.size(); ++index) {
                const auto ref = issue_order[index];
                auto& session = ref.side == Side::Left
                    ? left_session : right_session;
                const auto input = ref.side == Side::Left
                    ? left.get_column(ref.ordinal)
                    : right.get_column(ref.ordinal);
                const auto& output = session->output(
                    static_cast<std::size_t>(ref.ordinal));
                gamax::exec::PhaseMetadata exchange_metadata{
                    "join.column.ce_pull_exchange",
                    gamax::exec::PhaseCategory::Communication,
                    gamax::exec::CommunicationPattern::Shuffle,
                    "cuda_memcpy_batch_pull",
                    logical_column_bytes(input),
                    logical_column_bytes(output.view()),
                    input.size(), output.size(), context.size - 1};
                set_column_identity(exchange_metadata, ref);
                ProfilePhase exchange_phase(
                    profile_sink, std::move(exchange_metadata),
                    transfer_stream);
                session->enqueue_pull(
                    static_cast<std::size_t>(ref.ordinal));
                exchange_phase.finish();
            }
            record_event(payload_exchange_stop, transfer_stream);
            exchange_range.finish();
            alps::collectives::finish_ce_pull_full_staging_batch(context);
            payload_pulls_enqueued = true;
        };

    record_event(match_start, runtime.join_stream);
    ProfilePhase match_phase(
        profile_sink,
        {"join.column.match", gamax::exec::PhaseCategory::Compute},
        runtime.join_stream);
    NvtxRange match_range("column_overlap.ce_pull.match_keys");
    auto matches = gamax::join::eq_join_indices(
        input_for({Side::Left, left_key}),
        input_for({Side::Right, right_key}), hint, runtime.join_stream,
        release_payload_pulls);
    record_event(match_stop, runtime.join_stream);
    match_phase.finish();
    match_range.finish();
    if (!payload_pulls_enqueued) {
        throw std::logic_error(
            "column_overlap_eq_join: hash join did not release payload pulls");
    }

    for (std::size_t index = key_issue_count;
         index < issue_order.size(); ++index) {
        release_received(issue_order[index]);
    }

    const auto output_columns = static_cast<std::size_t>(
        1 + left.num_columns() - 1 + right.num_columns() - 1);
    std::vector<gamax::column_ptr> output(output_columns);
    auto left_output_slot = [left_key](size_type ordinal) -> std::size_t {
        return static_cast<std::size_t>(
            1 + ordinal - (ordinal > left_key ? 1 : 0));
    };
    const auto right_base = static_cast<std::size_t>(left.num_columns());
    auto right_output_slot = [right_key, right_base](size_type ordinal) {
        return right_base + static_cast<std::size_t>(
            ordinal - (ordinal > right_key ? 1 : 0));
    };
    auto output_slot_for = [&] (ColumnRef ref) {
        return ref.side == Side::Left
            ? left_output_slot(ref.ordinal)
            : right_output_slot(ref.ordinal);
    };
    auto match_indices_for = [&] (ColumnRef ref) -> gamax::column_view {
        return ref.side == Side::Left
            ? matches.left_indices->view() : matches.right_indices->view();
    };
    auto retire_received = [&] (ColumnRef ref) {
        const bool partitioned = ref.side == Side::Left
            ? partition_left : partition_right;
        if (!partitioned) return;
        auto& received = ref.side == Side::Left
            ? left_recv[static_cast<std::size_t>(ref.ordinal)]
            : right_recv[static_cast<std::size_t>(ref.ordinal)];
        set_column_lifetime_stream(*received, runtime.join_stream);
        if (early_reclaim) received.reset();
    };

    record_event(materialize_start, runtime.join_stream);
    ProfilePhase materialize_phase(
        profile_sink,
        {"join.column.materialize", gamax::exec::PhaseCategory::Compute},
        runtime.join_stream);
    NvtxRange materialize_range(
        "column_overlap.ce_pull.materialize_payloads");
    output[0] = gamax::join::detail::simple_materialize(
        input_for({Side::Left, left_key}), matches.left_indices->view(),
        runtime.join_stream);
    retire_received({Side::Left, left_key});

    std::vector<StringOutputState> string_outputs;
    for (const auto ref : payloads) {
        wait_for(ref);
        const auto input = input_for(ref);
        if (input.type().id() == gamax::type_id::STRING) {
            string_outputs.push_back(StringOutputState{
                ref,
                gamax::plan_gather_strings(
                    gamax::string_column_view(input, gamax::ARROW_STRING),
                    match_indices_for(ref), runtime.join_stream)});
        } else {
            output[output_slot_for(ref)] =
                gamax::join::detail::simple_materialize(
                    input, match_indices_for(ref), runtime.join_stream);
            retire_received(ref);
        }
    }

    PinnedCounts output_string_bytes(string_outputs.size());
    Event string_plans_ready(!string_outputs.empty());
    for (std::size_t index = 0; index < string_outputs.size(); ++index) {
        auto& state = string_outputs[index];
        ALPS_CUDACHECK(cudaMemcpyAsync(
            output_string_bytes.data() + index,
            state.gather_plan.offsets->data<size_type>() +
                state.gather_plan.offsets->size() - 1,
            sizeof(size_type), cudaMemcpyDeviceToHost,
            runtime.join_stream));
        state.gather_plan.lengths->buffer().set_stream(runtime.join_stream);
        state.gather_plan.lengths.reset();
    }
    if (!string_outputs.empty()) {
        ALPS_CUDACHECK(cudaEventRecord(
            string_plans_ready.value, runtime.join_stream));
        ALPS_CUDACHECK(cudaEventSynchronize(string_plans_ready.value));
    }
    for (std::size_t index = 0; index < string_outputs.size(); ++index) {
        auto& state = string_outputs[index];
        auto result = gamax::materialize_gathered_strings(
            gamax::string_column_view(
                input_for(state.ref), gamax::ARROW_STRING),
            match_indices_for(state.ref),
            std::move(state.gather_plan.offsets),
            output_string_bytes[index], runtime.join_stream);
        output[output_slot_for(state.ref)] =
            gamax::column_ptr(result.release());
        retire_received(state.ref);
    }

    retire_received({Side::Right, right_key});
    record_event(materialize_stop, runtime.join_stream);
    materialize_phase.finish();
    materialize_range.finish();
    ALPS_CUDACHECK(cudaEventRecord(final_event, runtime.join_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(caller_stream, final_event, 0));
    for (auto& column : output) {
        set_column_lifetime_stream(*column, caller_stream);
    }
    record_event(total_stop, caller_stream);

    if (timing) {
        ALPS_CUDACHECK(cudaEventSynchronize(total_stop.value));
        timing->total_ms = elapsed(total_start.value, total_stop.value);
        timing->key_prepare_ms = elapsed(
            key_prepare_start.value, key_prepare_stop.value);
        timing->payload_prepare_ms = elapsed(
            payload_prepare_start.value, payload_prepare_stop.value);
        timing->key_exchange_ms = elapsed(
            key_exchange_start.value, key_exchange_stop.value);
        timing->payload_exchange_ms = elapsed(
            payload_exchange_start.value, payload_exchange_stop.value);
        timing->match_ms = elapsed(match_start.value, match_stop.value);
        timing->materialize_ms = elapsed(
            materialize_start.value, materialize_stop.value);
    }
    return gamax::make_table(std::move(output));
}

} // namespace

std::unique_ptr<gamax::table> column_overlap_eq_join(
    gamax::table_view left,
    size_type left_key,
    gamax::table_view right,
    size_type right_key,
    const gamax::join::join_hint& hint,
    const alps::Context& context,
    cudaStream_t caller_stream,
    bool partition_left,
    bool partition_right,
    ColumnSchedule schedule,
    ColumnOverlapTiming* timing,
    gamax::exec::StepProfileSink* profile_sink,
    bool communication_high_priority,
    bool early_reclaim,
    alps::collectives::Backend exchange_backend,
    alps::collectives::CePullSchedule ce_pull_schedule,
    int ce_pull_slots)
{
    if (exchange_backend == alps::collectives::Backend::CE_PULL) {
        return column_overlap_eq_join_ce_pull(
            left, left_key, right, right_key, hint, context, caller_stream,
            partition_left, partition_right, schedule, timing, profile_sink,
            communication_high_priority, early_reclaim, ce_pull_schedule,
            ce_pull_slots);
    }
    if (exchange_backend != alps::collectives::Backend::NCCL) {
        throw std::invalid_argument(
            "column_overlap_eq_join: exchange backend must be NCCL or CE_PULL");
    }
    const auto left_columns = static_cast<std::size_t>(left.num_columns());
    const auto right_columns = static_cast<std::size_t>(right.num_columns());
    Event total_start(timing != nullptr, true);
    Event total_stop(timing != nullptr, true);
    Event key_prepare_start(timing != nullptr, true);
    Event key_prepare_stop(timing != nullptr, true);
    Event payload_prepare_start(timing != nullptr, true);
    Event payload_prepare_stop(timing != nullptr, true);
    Event key_exchange_start(timing != nullptr, true);
    Event key_exchange_stop(timing != nullptr, true);
    Event payload_exchange_start(timing != nullptr, true);
    Event payload_exchange_stop(timing != nullptr, true);
    Event match_start(timing != nullptr, true);
    Event match_stop(timing != nullptr, true);
    Event materialize_start(timing != nullptr, true);
    Event materialize_stop(timing != nullptr, true);

    record_event(total_start, caller_stream);
    alps::collectives::HashPartitionPlan left_plan;
    alps::collectives::HashPartitionPlan right_plan;
    ProfilePhase partition_phase(
        profile_sink,
        {"join.column.partition_plan", gamax::exec::PhaseCategory::Compute},
        caller_stream);
    std::exception_ptr local_error;
    try {
        validate_local(left, left_key, right, right_key, hint);
        if (!context.column_join_runtime) {
            context.column_join_runtime = std::make_shared<ColumnJoinRuntime>(
                communication_high_priority);
        } else if (context.column_join_runtime->communication_high_priority !=
                   communication_high_priority) {
            throw std::invalid_argument(
                "column_overlap_eq_join: communication priority cannot change "
                "within one Context");
        }
        context.column_join_runtime->ensure_dependency_events(
            4 + 4 * (left_columns + right_columns));
        if (partition_left) {
            left_plan = alps::collectives::make_hash_partition_plan(
                left.select_columns({left_key}), context, caller_stream);
        }
        if (partition_right) {
            right_plan = alps::collectives::make_hash_partition_plan(
                right.select_columns({right_key}), context, caller_stream);
        }
    } catch (...) {
        local_error = std::current_exception();
    }
    partition_phase.finish();
    require_rank_wide_success(!local_error, context, "validation and planning");
    if (local_error) std::rethrow_exception(local_error);

    auto& runtime = *context.column_join_runtime;
    std::size_t dependency_index = 0;
    const auto entry = runtime.dependency_events[dependency_index++];
    const auto final_event = runtime.dependency_events[dependency_index++];
    const auto all_prepared = runtime.dependency_events[dependency_index++];
    const auto string_output_plans_ready =
        runtime.dependency_events[dependency_index++];
    std::vector<cudaEvent_t> left_prepared(left_columns);
    std::vector<cudaEvent_t> right_prepared(right_columns);
    std::vector<cudaEvent_t> left_received(left_columns);
    std::vector<cudaEvent_t> right_received(right_columns);
    std::vector<cudaEvent_t> left_lengths_received(left_columns);
    std::vector<cudaEvent_t> right_lengths_received(right_columns);
    std::vector<cudaEvent_t> left_offsets_ready(left_columns);
    std::vector<cudaEvent_t> right_offsets_ready(right_columns);
    for (auto& event : left_prepared) {
        event = runtime.dependency_events[dependency_index++];
    }
    for (auto& event : right_prepared) {
        event = runtime.dependency_events[dependency_index++];
    }
    for (auto& event : left_received) {
        event = runtime.dependency_events[dependency_index++];
    }
    for (auto& event : right_received) {
        event = runtime.dependency_events[dependency_index++];
    }
    for (auto& event : left_lengths_received) {
        event = runtime.dependency_events[dependency_index++];
    }
    for (auto& event : right_lengths_received) {
        event = runtime.dependency_events[dependency_index++];
    }
    for (auto& event : left_offsets_ready) {
        event = runtime.dependency_events[dependency_index++];
    }
    for (auto& event : right_offsets_ready) {
        event = runtime.dependency_events[dependency_index++];
    }
    ALPS_CUDACHECK(cudaEventRecord(entry, caller_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.preparation_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.communication_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(runtime.join_stream, entry, 0));

    ProfilePhase key_prepare_phase;
    ProfilePhase payload_prepare_phase;
    if (partition_left) {
        alps::collectives::complete_receive_pattern(left_plan.pattern, context);
    }
    if (partition_right) {
        alps::collectives::complete_receive_pattern(right_plan.pattern, context);
    }

    std::vector<gamax::column_ptr> left_send(left_columns);
    std::vector<gamax::column_ptr> right_send(right_columns);
    std::vector<gamax::column_ptr> left_recv(left_columns);
    std::vector<gamax::column_ptr> right_recv(right_columns);

    std::vector<ColumnRef> issue_order;
    issue_order.reserve(left_columns + right_columns);
    if (partition_left) issue_order.push_back({Side::Left, left_key});
    if (partition_right) issue_order.push_back({Side::Right, right_key});
    const auto key_issue_count = issue_order.size();
    auto payloads = payload_order(left, left_key, right, right_key, schedule);
    const bool has_string_payload = std::any_of(
        payloads.begin(), payloads.end(), [&] (ColumnRef ref) {
            return (ref.side == Side::Left
                    ? left.get_column(ref.ordinal)
                    : right.get_column(ref.ordinal)).type().id() ==
                gamax::type_id::STRING;
        });
    for (const auto payload : payloads) {
        if ((payload.side == Side::Left && partition_left) ||
            (payload.side == Side::Right && partition_right)) {
            issue_order.push_back(payload);
        }
    }

    constexpr auto no_string = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> left_string_state(left_columns, no_string);
    std::vector<std::size_t> right_string_state(right_columns, no_string);
    std::vector<StringExchangeState> string_states;
    for (std::size_t index = key_issue_count; index < issue_order.size(); ++index) {
        const auto ref = issue_order[index];
        const auto input = ref.side == Side::Left
            ? left.get_column(ref.ordinal)
            : right.get_column(ref.ordinal);
        if (input.type().id() != gamax::type_id::STRING) continue;
        const auto state_index = string_states.size();
        string_states.push_back(StringExchangeState{ref});
        if (ref.side == Side::Left) {
            left_string_state[static_cast<std::size_t>(ref.ordinal)] = state_index;
        } else {
            right_string_state[static_cast<std::size_t>(ref.ordinal)] = state_index;
        }
    }
    std::unique_ptr<PinnedCounts> send_char_boundaries;

    auto string_state_for = [&] (ColumnRef ref) -> StringExchangeState& {
        const auto index = ref.side == Side::Left
            ? left_string_state[static_cast<std::size_t>(ref.ordinal)]
            : right_string_state[static_cast<std::size_t>(ref.ordinal)];
        if (index == no_string) {
            throw std::logic_error("missing ColumnOverlap STRING exchange state");
        }
        return string_states[index];
    };

    local_error = nullptr;
    try {
        send_char_boundaries = std::make_unique<PinnedCounts>(
            string_states.size() * static_cast<std::size_t>(context.size) * 2);
        if (partition_left) {
            for (size_type column = 0; column < left.num_columns(); ++column) {
                if (left.get_column(column).type().id() != gamax::type_id::STRING) {
                    left_recv[static_cast<std::size_t>(column)] = gamax::make_column(
                        left.get_column(column).type(), left_plan.pattern.total_recv(),
                        runtime.communication_stream);
                }
            }
        }
        if (partition_right) {
            for (size_type column = 0; column < right.num_columns(); ++column) {
                if (right.get_column(column).type().id() != gamax::type_id::STRING) {
                    right_recv[static_cast<std::size_t>(column)] = gamax::make_column(
                        right.get_column(column).type(), right_plan.pattern.total_recv(),
                        runtime.communication_stream);
                }
            }
        }
        for (auto& state : string_states) {
            const auto recv_rows = state.ref.side == Side::Left
                ? left_plan.pattern.total_recv()
                : right_plan.pattern.total_recv();
            state.recv_lengths = gamax::make_column<int32_t>(
                recv_rows, runtime.communication_stream);
        }

        record_event(key_prepare_start, runtime.preparation_stream);
        key_prepare_phase.start(
            profile_sink,
            {"join.column.key_prepare", gamax::exec::PhaseCategory::Compute},
            runtime.preparation_stream);
        bool payload_prepare_started = false;
        if (key_issue_count == 0) {
            record_event(key_prepare_stop, runtime.preparation_stream);
            record_event(payload_prepare_start, runtime.preparation_stream);
            key_prepare_phase.finish();
            payload_prepare_phase.start(
                profile_sink,
                {"join.column.payload_prepare", gamax::exec::PhaseCategory::Compute},
                runtime.preparation_stream);
            payload_prepare_started = true;
        }
        for (std::size_t index = 0; index < issue_order.size(); ++index) {
            if (index == key_issue_count) {
                record_event(key_prepare_stop, runtime.preparation_stream);
                record_event(payload_prepare_start, runtime.preparation_stream);
                key_prepare_phase.finish();
                payload_prepare_phase.start(
                    profile_sink,
                    {"join.column.payload_prepare",
                     gamax::exec::PhaseCategory::Compute},
                    runtime.preparation_stream);
                payload_prepare_started = true;
            }
            const auto ref = issue_order[index];
            auto& send = ref.side == Side::Left
                ? left_send[static_cast<std::size_t>(ref.ordinal)]
                : right_send[static_cast<std::size_t>(ref.ordinal)];
            const auto input = ref.side == Side::Left
                ? left.get_column(ref.ordinal)
                : right.get_column(ref.ordinal);
            const auto& plan = ref.side == Side::Left ? left_plan : right_plan;
            if (input.type().id() == gamax::type_id::STRING) {
                auto& state = string_state_for(ref);
                auto gather_plan = gamax::plan_gather_strings(
                    gamax::string_column_view(input, gamax::ARROW_STRING),
                    plan.permutation->view(), runtime.preparation_stream);
                state.send_lengths = std::move(gather_plan.lengths);
                auto string_send = gamax::materialize_gathered_strings(
                    gamax::string_column_view(input, gamax::ARROW_STRING),
                    plan.permutation->view(), std::move(gather_plan.offsets),
                    input.auxiliary_bytes(), runtime.preparation_stream);
                send = gamax::column_ptr(string_send.release());

                const auto state_index = static_cast<std::size_t>(&state - string_states.data());
                const auto* offsets = static_cast<const size_type*>(
                    send->view().auxiliary_data());
                for (int peer = 0; peer < context.size; ++peer) {
                    const auto begin_row = plan.pattern.send_offsets[peer];
                    const auto end_row = begin_row + plan.pattern.send_counts[peer];
                    const auto boundary_base =
                        (state_index * static_cast<std::size_t>(context.size) +
                         static_cast<std::size_t>(peer)) * 2;
                    ALPS_CUDACHECK(cudaMemcpyAsync(
                        send_char_boundaries->data() + boundary_base,
                        offsets + begin_row, sizeof(size_type),
                        cudaMemcpyDeviceToHost, runtime.preparation_stream));
                    ALPS_CUDACHECK(cudaMemcpyAsync(
                        send_char_boundaries->data() + boundary_base + 1,
                        offsets + end_row, sizeof(size_type),
                        cudaMemcpyDeviceToHost, runtime.preparation_stream));
                }
            } else {
                send = alps::collectives::apply_partition(
                    input, plan,
                    runtime.preparation_stream);
            }
            if (ref.side == Side::Left) {
                ALPS_CUDACHECK(cudaEventRecord(
                    left_prepared[static_cast<std::size_t>(ref.ordinal)],
                    runtime.preparation_stream));
            } else {
                ALPS_CUDACHECK(cudaEventRecord(
                    right_prepared[static_cast<std::size_t>(ref.ordinal)],
                    runtime.preparation_stream));
            }
        }
        if (!payload_prepare_started) {
            record_event(key_prepare_stop, runtime.preparation_stream);
            record_event(payload_prepare_start, runtime.preparation_stream);
            key_prepare_phase.finish();
            payload_prepare_phase.start(
                profile_sink,
                {"join.column.payload_prepare", gamax::exec::PhaseCategory::Compute},
                runtime.preparation_stream);
        }
        record_event(payload_prepare_stop, runtime.preparation_stream);
        ALPS_CUDACHECK(cudaEventRecord(all_prepared, runtime.preparation_stream));
        payload_prepare_phase.finish();
    } catch (...) {
        key_prepare_phase.finish();
        payload_prepare_phase.finish();
        local_error = std::current_exception();
    }
    require_rank_wide_success(!local_error, context, "column allocation");
    if (local_error) {
        cudaStreamSynchronize(runtime.preparation_stream);
        std::rethrow_exception(local_error);
    }

    // The saved permutations are read only by the preparation stream from this
    // point onward; make their eventual deallocation stream-safe.
    if (partition_left) {
        left_plan.permutation->buffer().set_stream(runtime.preparation_stream);
    }
    if (partition_right) {
        right_plan.permutation->buffer().set_stream(runtime.preparation_stream);
    }

    auto exchange_metadata = [&] (
        const char* name, std::size_t begin, std::size_t end) {
        gamax::exec::PhaseMetadata metadata{
            name,
            gamax::exec::PhaseCategory::Communication,
            gamax::exec::CommunicationPattern::Shuffle,
            "nccl_send_recv",
            0, 0, 0, 0, 0};
        bool includes_left = false;
        bool includes_right = false;
        for (std::size_t index = begin; index < end; ++index) {
            const auto ref = issue_order[index];
            if (ref.side == Side::Left) {
                includes_left = true;
                metadata.bytes_sent += logical_column_bytes(
                    left.get_column(ref.ordinal));
                metadata.bytes_received +=
                    left_recv[static_cast<std::size_t>(ref.ordinal)]
                    ? logical_column_bytes(
                          left_recv[static_cast<std::size_t>(ref.ordinal)]->view())
                    : 0;
            } else {
                includes_right = true;
                metadata.bytes_sent += logical_column_bytes(
                    right.get_column(ref.ordinal));
                metadata.bytes_received +=
                    right_recv[static_cast<std::size_t>(ref.ordinal)]
                    ? logical_column_bytes(
                          right_recv[static_cast<std::size_t>(ref.ordinal)]->view())
                    : 0;
            }
        }
        // Rows describe the logical table fragments represented by this phase,
        // not the number of per-column NCCL calls. Count each participating
        // side once even when the phase exchanges several payload columns.
        if (includes_left) {
            metadata.rows_sent += left.num_rows();
            metadata.rows_received += left_plan.pattern.total_recv();
        }
        if (includes_right) {
            metadata.rows_sent += right.num_rows();
            metadata.rows_received += right_plan.pattern.total_recv();
        }
        if (includes_left || includes_right) {
            metadata.peer_count = context.size - 1;
        }
        return metadata;
    };
    NvtxRange exchange_range("column_overlap.exchange.initial");
    record_event(key_exchange_start, runtime.communication_stream);
    for (std::size_t index = 0; index < key_issue_count; ++index) {
        const auto ref = issue_order[index];
        auto metadata = exchange_metadata(
            "join.column.key_exchange", index, index + 1);
        set_column_identity(metadata, ref);
        ProfilePhase key_exchange_phase(
            profile_sink, std::move(metadata),
            runtime.communication_stream);
        if (ref.side == Side::Left) {
            auto& send = left_send[static_cast<std::size_t>(ref.ordinal)];
            auto& recv = left_recv[static_cast<std::size_t>(ref.ordinal)];
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                runtime.communication_stream,
                left_prepared[static_cast<std::size_t>(ref.ordinal)], 0));
            alps::collectives::alltoall_column(
                send->view(), recv->mutable_view(), left_plan.pattern,
                context, runtime.communication_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                left_received[static_cast<std::size_t>(ref.ordinal)],
                runtime.communication_stream));
            send->buffer().set_stream(runtime.communication_stream);
            if (early_reclaim) send.reset();
        } else {
            auto& send = right_send[static_cast<std::size_t>(ref.ordinal)];
            auto& recv = right_recv[static_cast<std::size_t>(ref.ordinal)];
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                runtime.communication_stream,
                right_prepared[static_cast<std::size_t>(ref.ordinal)], 0));
            alps::collectives::alltoall_column(
                send->view(), recv->mutable_view(), right_plan.pattern,
                context, runtime.communication_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                right_received[static_cast<std::size_t>(ref.ordinal)],
                runtime.communication_stream));
            send->buffer().set_stream(runtime.communication_stream);
            if (early_reclaim) send.reset();
        }
    }
    record_event(key_exchange_stop, runtime.communication_stream);
    record_event(payload_exchange_start, runtime.communication_stream);

    for (std::size_t index = key_issue_count; index < issue_order.size(); ++index) {
        const auto ref = issue_order[index];
        const auto send_rows = ref.side == Side::Left
            ? left.num_rows() : right.num_rows();
        const auto recv_rows = ref.side == Side::Left
            ? left_plan.pattern.total_recv() : right_plan.pattern.total_recv();
        auto& send = ref.side == Side::Left
            ? left_send[static_cast<std::size_t>(ref.ordinal)]
            : right_send[static_cast<std::size_t>(ref.ordinal)];
        const auto& plan = ref.side == Side::Left ? left_plan : right_plan;
        const auto prepared = ref.side == Side::Left
            ? left_prepared[static_cast<std::size_t>(ref.ordinal)]
            : right_prepared[static_cast<std::size_t>(ref.ordinal)];
        ALPS_CUDACHECK(cudaStreamWaitEvent(
            runtime.communication_stream, prepared, 0));

        if (send->type().id() == gamax::type_id::STRING) {
            auto& state = string_state_for(ref);
            gamax::exec::PhaseMetadata metadata{
                "join.column.string_lengths_exchange",
                gamax::exec::PhaseCategory::Communication,
                gamax::exec::CommunicationPattern::Shuffle,
                "nccl_send_recv",
                static_cast<std::int64_t>(send_rows) *
                    static_cast<std::int64_t>(sizeof(int32_t)),
                static_cast<std::int64_t>(recv_rows) *
                    static_cast<std::int64_t>(sizeof(int32_t)),
                send_rows, recv_rows, context.size - 1};
            set_column_identity(metadata, ref);
            ProfilePhase phase(
                profile_sink, std::move(metadata),
                runtime.communication_stream);
            alps::collectives::alltoall_column(
                state.send_lengths->view(), state.recv_lengths->mutable_view(),
                plan.pattern, context, runtime.communication_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                ref.side == Side::Left
                    ? left_lengths_received[static_cast<std::size_t>(ref.ordinal)]
                    : right_lengths_received[static_cast<std::size_t>(ref.ordinal)],
                runtime.communication_stream));
            state.send_lengths->buffer().set_stream(runtime.communication_stream);
            state.send_lengths.reset();
        } else {
            auto& recv = ref.side == Side::Left
                ? left_recv[static_cast<std::size_t>(ref.ordinal)]
                : right_recv[static_cast<std::size_t>(ref.ordinal)];
            gamax::exec::PhaseMetadata metadata{
                "join.column.payload_exchange",
                gamax::exec::PhaseCategory::Communication,
                gamax::exec::CommunicationPattern::Shuffle,
                "nccl_send_recv",
                logical_column_bytes(send->view()),
                logical_column_bytes(recv->view()),
                send_rows, recv_rows, context.size - 1};
            set_column_identity(metadata, ref);
            ProfilePhase phase(
                profile_sink, std::move(metadata),
                runtime.communication_stream);
            alps::collectives::alltoall_column(
                send->view(), recv->mutable_view(), plan.pattern,
                context, runtime.communication_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                ref.side == Side::Left
                    ? left_received[static_cast<std::size_t>(ref.ordinal)]
                    : right_received[static_cast<std::size_t>(ref.ordinal)],
                runtime.communication_stream));
            send->buffer().set_stream(runtime.communication_stream);
            if (early_reclaim) send.reset();
        }
    }

    const auto output_columns = static_cast<std::size_t>(
        1 + left.num_columns() - 1 + right.num_columns() - 1);
    gamax::join::EqJoinMatches matches;
    std::vector<gamax::column_ptr> output;
    ProfilePhase match_phase;
    ProfilePhase materialize_phase;
    std::unique_ptr<NvtxRange> match_range;
    std::unique_ptr<NvtxRange> materialize_range;

    auto left_output_slot = [left_key](size_type ordinal) -> std::size_t {
        return static_cast<std::size_t>(
            1 + ordinal - (ordinal > left_key ? 1 : 0));
    };
    const auto right_base = static_cast<std::size_t>(left.num_columns());
    auto right_output_slot = [right_key, right_base](size_type ordinal) {
        return right_base + static_cast<std::size_t>(
            ordinal - (ordinal > right_key ? 1 : 0));
    };

    auto input_for = [&] (ColumnRef ref) -> gamax::column_view {
        if (ref.side == Side::Left) {
            return partition_left
                ? left_recv[static_cast<std::size_t>(ref.ordinal)]->view()
                : left.get_column(ref.ordinal);
        }
        return partition_right
            ? right_recv[static_cast<std::size_t>(ref.ordinal)]->view()
            : right.get_column(ref.ordinal);
    };

    auto match_indices_for = [&] (ColumnRef ref) -> gamax::column_view {
        return ref.side == Side::Left
            ? matches.left_indices->view()
            : matches.right_indices->view();
    };

    auto output_slot_for = [&] (ColumnRef ref) {
        return ref.side == Side::Left
            ? left_output_slot(ref.ordinal)
            : right_output_slot(ref.ordinal);
    };

    local_error = nullptr;
    try {
        inject_test_failure(context, "string_offsets");
        for (auto& state : string_states) {
            const auto ordinal = static_cast<std::size_t>(state.ref.ordinal);
            const auto lengths_received = state.ref.side == Side::Left
                ? left_lengths_received[ordinal] : right_lengths_received[ordinal];
            const auto offsets_ready = state.ref.side == Side::Left
                ? left_offsets_ready[ordinal] : right_offsets_ready[ordinal];
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                runtime.preparation_stream, lengths_received, 0));
            gamax::exec::PhaseMetadata metadata{
                "join.column.string_offsets_rebuild",
                gamax::exec::PhaseCategory::Compute};
            set_column_identity(metadata, state.ref);
            ProfilePhase phase(
                profile_sink, std::move(metadata),
                runtime.preparation_stream);
            state.recv_offsets = gamax::util::device::prefix_sum(
                state.recv_lengths->view(),
                gamax::prefix_sum_options::INCLUSIVE_PAD_ZERO,
                gamax::data_type{gamax::type_id::INT64},
                runtime.preparation_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                offsets_ready, runtime.preparation_stream));
            state.recv_lengths->buffer().set_stream(runtime.preparation_stream);
            state.recv_lengths.reset();
        }

        if (partition_left) {
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                runtime.join_stream,
                left_received[static_cast<std::size_t>(left_key)], 0));
        }
        if (partition_right) {
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                runtime.join_stream,
                right_received[static_cast<std::size_t>(right_key)], 0));
        }
        record_event(match_start, runtime.join_stream);
        match_phase.start(
            profile_sink,
            {"join.column.match", gamax::exec::PhaseCategory::Compute},
            runtime.join_stream);
        match_range =
            std::make_unique<NvtxRange>("column_overlap.match_keys");
        matches = gamax::join::eq_join_indices(
            partition_left
                ? left_recv[static_cast<std::size_t>(left_key)]->view()
                : left.get_column(left_key),
            partition_right
                ? right_recv[static_cast<std::size_t>(right_key)]->view()
                : right.get_column(right_key),
            hint, runtime.join_stream);
        record_event(match_stop, runtime.join_stream);
        match_phase.finish();
        match_range->finish();

        output.resize(output_columns);
        record_event(materialize_start, runtime.join_stream);
        materialize_phase.start(
            profile_sink,
            {"join.column.materialize", gamax::exec::PhaseCategory::Compute},
            runtime.join_stream);
        materialize_range = std::make_unique<NvtxRange>(
            "column_overlap.materialize_payloads");

        output[0] = gamax::join::detail::simple_materialize(
            partition_left
                ? left_recv[static_cast<std::size_t>(left_key)]->view()
                : left.get_column(left_key),
            matches.left_indices->view(),
            runtime.join_stream);
        if (partition_left) {
            left_recv[static_cast<std::size_t>(left_key)]->buffer().set_stream(
                runtime.join_stream);
            if (early_reclaim) {
                left_recv[static_cast<std::size_t>(left_key)].reset();
            }
        }

        // Readiness-aware materialization: enqueue every fixed-width output
        // before coordinating variable-length character buffers.
        for (const auto ref : payloads) {
            const auto original = ref.side == Side::Left
                ? left.get_column(ref.ordinal) : right.get_column(ref.ordinal);
            if (original.type().id() == gamax::type_id::STRING) continue;
            const bool partitioned = ref.side == Side::Left
                ? partition_left : partition_right;
            if (partitioned) {
                ALPS_CUDACHECK(cudaStreamWaitEvent(
                    runtime.join_stream,
                    ref.side == Side::Left
                        ? left_received[static_cast<std::size_t>(ref.ordinal)]
                        : right_received[static_cast<std::size_t>(ref.ordinal)],
                    0));
            }
            output[output_slot_for(ref)] =
                gamax::join::detail::simple_materialize(
                    input_for(ref), match_indices_for(ref), runtime.join_stream);
            if (partitioned) {
                auto& received = ref.side == Side::Left
                    ? left_recv[static_cast<std::size_t>(ref.ordinal)]
                    : right_recv[static_cast<std::size_t>(ref.ordinal)];
                set_column_lifetime_stream(*received, runtime.join_stream);
                if (early_reclaim) received.reset();
            }
        }
    } catch (...) {
        match_phase.finish();
        materialize_phase.finish();
        if (match_range) match_range->finish();
        if (materialize_range) materialize_range->finish();
        local_error = std::current_exception();
    }
    exchange_range.finish();

    if (has_string_payload) {
        if (!rank_wide_stage_succeeded(!local_error, context)) {
            materialize_phase.finish();
            if (materialize_range) materialize_range->finish();
            drain_runtime(runtime);
            throw_rank_wide_stage_failure(
                local_error, "STRING offset and match planning");
        }
    } else if (local_error) {
        drain_runtime(runtime);
        std::rethrow_exception(local_error);
    }

    // Resolve every STRING column's per-peer character counts at one batched
    // coordination point. Row lengths and all fixed-width payload transfers
    // were already enqueued, so this host work overlaps the key match and fixed
    // materialization work above.
    if (!string_states.empty()) {
        NvtxRange metadata_range("column_overlap.string_count_coordination");
        const auto strings = string_states.size();
        const auto world = static_cast<std::size_t>(context.size);
        std::vector<size_type> batched_send_counts;
        std::vector<size_type> batched_recv_counts;
        bool valid_boundaries =
            strings <= static_cast<std::size_t>(std::numeric_limits<int>::max());
        local_error = nullptr;
        try {
            ALPS_CUDACHECK(cudaEventSynchronize(all_prepared));
            batched_send_counts.assign(world * strings, 0);
            batched_recv_counts.assign(world * strings, 0);
            for (std::size_t string = 0; string < strings; ++string) {
                const auto ref = string_states[string].ref;
                const auto& send = ref.side == Side::Left
                    ? left_send[static_cast<std::size_t>(ref.ordinal)]
                    : right_send[static_cast<std::size_t>(ref.ordinal)];
                const auto available = static_cast<size_type>(
                    send->view().auxiliary_bytes());
                for (std::size_t peer = 0; peer < world; ++peer) {
                    const auto boundary_base = (string * world + peer) * 2;
                    const auto begin = (*send_char_boundaries)[boundary_base];
                    const auto end = (*send_char_boundaries)[boundary_base + 1];
                    valid_boundaries = valid_boundaries &&
                        begin >= 0 && end >= begin && end <= available;
                    batched_send_counts[peer * strings + string] = end - begin;
                    string_states[string].char_pattern.send_offsets.push_back(begin);
                    string_states[string].char_pattern.send_counts.push_back(end - begin);
                }
            }
        } catch (...) {
            local_error = std::current_exception();
        }
        if (!rank_wide_stage_succeeded(
                !local_error && valid_boundaries, context)) {
            metadata_range.finish();
            materialize_phase.finish();
            if (materialize_range) materialize_range->finish();
            drain_runtime(runtime);
            if (local_error) std::rethrow_exception(local_error);
            throw_rank_wide_stage_failure(
                nullptr, "STRING character-count planning");
        }
        ALPS_MPICHECK(MPI_Alltoall(
            batched_send_counts.data(), static_cast<int>(strings),
            alps::convert::mpi_size_type(),
            batched_recv_counts.data(), static_cast<int>(strings),
            alps::convert::mpi_size_type(), context.mpi_comm));

        local_error = nullptr;
        try {
            for (std::size_t string = 0; string < strings; ++string) {
                auto& state = string_states[string];
                auto& pattern = state.char_pattern;
                pattern.recv_counts.resize(world);
                pattern.recv_offsets.resize(world);
                size_type recv_bytes = 0;
                for (std::size_t peer = 0; peer < world; ++peer) {
                    const auto count =
                        batched_recv_counts[peer * strings + string];
                    if (count < 0 ||
                        recv_bytes > std::numeric_limits<size_type>::max() - count) {
                        throw std::overflow_error(
                            "column_overlap_eq_join: invalid STRING receive size");
                    }
                    pattern.recv_counts[peer] = count;
                    pattern.recv_offsets[peer] = recv_bytes;
                    recv_bytes += pattern.recv_counts[peer];
                }
                gamax::device_buffer recv_chars(
                    static_cast<std::size_t>(recv_bytes),
                    runtime.communication_stream);
                auto recv_string = gamax::make_string_column(
                    std::move(recv_chars),
                    std::move(state.recv_offsets->buffer()),
                    state.ref.side == Side::Left
                        ? left_plan.pattern.total_recv()
                        : right_plan.pattern.total_recv(),
                    runtime.communication_stream, gamax::ARROW_STRING);
                auto& recv = state.ref.side == Side::Left
                    ? left_recv[static_cast<std::size_t>(state.ref.ordinal)]
                    : right_recv[static_cast<std::size_t>(state.ref.ordinal)];
                recv = gamax::column_ptr(recv_string.release());
            }
        } catch (...) {
            local_error = std::current_exception();
        }
        if (!rank_wide_stage_succeeded(!local_error, context)) {
            metadata_range.finish();
            materialize_phase.finish();
            if (materialize_range) materialize_range->finish();
            drain_runtime(runtime);
            throw_rank_wide_stage_failure(
                local_error, "STRING receive allocation");
        }

        metadata_range.finish();
        NvtxRange chars_range("column_overlap.exchange.string_chars");
        for (auto& state : string_states) {
            const auto ref = state.ref;
            auto& send = ref.side == Side::Left
                ? left_send[static_cast<std::size_t>(ref.ordinal)]
                : right_send[static_cast<std::size_t>(ref.ordinal)];
            auto& recv = ref.side == Side::Left
                ? left_recv[static_cast<std::size_t>(ref.ordinal)]
                : right_recv[static_cast<std::size_t>(ref.ordinal)];
            const auto send_view = send->view();
            auto recv_view = recv->mutable_view();
            gamax::column_view send_chars(
                send_view.data<char>(),
                static_cast<size_type>(send_view.auxiliary_bytes()),
                gamax::data_type{gamax::type_id::INT8});
            gamax::mutable_column_view recv_chars(
                recv_view.data<char>(),
                static_cast<size_type>(recv_view.auxiliary_bytes()),
                gamax::data_type{gamax::type_id::INT8});
            const auto send_bytes = static_cast<std::int64_t>(send_chars.size());
            const auto recv_bytes = static_cast<std::int64_t>(recv_chars.size());
            gamax::exec::PhaseMetadata metadata{
                "join.column.string_chars_exchange",
                gamax::exec::PhaseCategory::Communication,
                gamax::exec::CommunicationPattern::Shuffle,
                "nccl_send_recv",
                send_bytes, recv_bytes,
                ref.side == Side::Left ? left.num_rows() : right.num_rows(),
                ref.side == Side::Left
                    ? left_plan.pattern.total_recv()
                    : right_plan.pattern.total_recv(),
                context.size - 1};
            set_column_identity(metadata, ref);
            ProfilePhase phase(
                profile_sink, std::move(metadata),
                runtime.communication_stream);
            alps::collectives::alltoall_column(
                send_chars, recv_chars, state.char_pattern,
                context, runtime.communication_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                ref.side == Side::Left
                    ? left_received[static_cast<std::size_t>(ref.ordinal)]
                    : right_received[static_cast<std::size_t>(ref.ordinal)],
                runtime.communication_stream));
            set_column_lifetime_stream(*send, runtime.communication_stream);
            if (early_reclaim) send.reset();
        }
        chars_range.finish();
    }
    record_event(payload_exchange_stop, runtime.communication_stream);

    // Plan all output STRING gathers first and read every exact output byte
    // count at one host synchronization. Only reconstructed offsets are needed
    // for this stage, so character transfers can continue concurrently.
    std::vector<StringOutputState> string_outputs;
    std::unique_ptr<PinnedCounts> output_string_bytes;
    local_error = nullptr;
    if (has_string_payload) {
        try {
            inject_test_failure(context, "string_output_plan");
            for (const auto ref : payloads) {
                const auto original = ref.side == Side::Left
                    ? left.get_column(ref.ordinal) : right.get_column(ref.ordinal);
                if (original.type().id() != gamax::type_id::STRING) continue;
                const bool partitioned = ref.side == Side::Left
                    ? partition_left : partition_right;
                if (partitioned) {
                    ALPS_CUDACHECK(cudaStreamWaitEvent(
                        runtime.join_stream,
                        ref.side == Side::Left
                            ? left_offsets_ready[
                                  static_cast<std::size_t>(ref.ordinal)]
                            : right_offsets_ready[
                                  static_cast<std::size_t>(ref.ordinal)],
                        0));
                }
                string_outputs.push_back(StringOutputState{
                    ref,
                    gamax::plan_gather_strings(
                        gamax::string_column_view(
                            input_for(ref), gamax::ARROW_STRING),
                        match_indices_for(ref), runtime.join_stream)});
            }

            output_string_bytes =
                std::make_unique<PinnedCounts>(string_outputs.size());
            for (std::size_t index = 0; index < string_outputs.size(); ++index) {
                auto& state = string_outputs[index];
                ALPS_CUDACHECK(cudaMemcpyAsync(
                    output_string_bytes->data() + index,
                    state.gather_plan.offsets->data<size_type>() +
                        state.gather_plan.offsets->size() - 1,
                    sizeof(size_type), cudaMemcpyDeviceToHost,
                    runtime.join_stream));
                state.gather_plan.lengths->buffer().set_stream(
                    runtime.join_stream);
                state.gather_plan.lengths.reset();
            }
            if (!string_outputs.empty()) {
                ALPS_CUDACHECK(cudaEventRecord(
                    string_output_plans_ready, runtime.join_stream));
                NvtxRange output_size_range(
                    "column_overlap.string_output_size_coordination");
                ALPS_CUDACHECK(cudaEventSynchronize(string_output_plans_ready));
            }
        } catch (...) {
            local_error = std::current_exception();
        }
        if (!rank_wide_stage_succeeded(!local_error, context)) {
            materialize_phase.finish();
            if (materialize_range) materialize_range->finish();
            drain_runtime(runtime);
            throw_rank_wide_stage_failure(
                local_error, "STRING output planning");
        }
    }

    local_error = nullptr;
    try {
        for (std::size_t index = 0; index < string_outputs.size(); ++index) {
            auto& state = string_outputs[index];
            const bool partitioned = state.ref.side == Side::Left
                ? partition_left : partition_right;
            if (partitioned) {
                ALPS_CUDACHECK(cudaStreamWaitEvent(
                    runtime.join_stream,
                    state.ref.side == Side::Left
                        ? left_received[static_cast<std::size_t>(state.ref.ordinal)]
                        : right_received[static_cast<std::size_t>(state.ref.ordinal)],
                    0));
            }
            auto result = gamax::materialize_gathered_strings(
                gamax::string_column_view(
                    input_for(state.ref), gamax::ARROW_STRING),
                match_indices_for(state.ref),
                std::move(state.gather_plan.offsets),
                (*output_string_bytes)[index], runtime.join_stream);
            output[output_slot_for(state.ref)] =
                gamax::column_ptr(result.release());
            if (partitioned) {
                auto& received = state.ref.side == Side::Left
                    ? left_recv[static_cast<std::size_t>(state.ref.ordinal)]
                    : right_recv[static_cast<std::size_t>(state.ref.ordinal)];
                set_column_lifetime_stream(*received, runtime.join_stream);
                if (early_reclaim) received.reset();
            }
        }
    } catch (...) {
        local_error = std::current_exception();
    }
    if (has_string_payload &&
        !rank_wide_stage_succeeded(!local_error, context)) {
        materialize_phase.finish();
        if (materialize_range) materialize_range->finish();
        drain_runtime(runtime);
        throw_rank_wide_stage_failure(
            local_error, "STRING output allocation");
    }

    // The right key is not materialized, but matching has completed before this
    // point, so it can safely retire on the join stream too.
    if (partition_right) {
        set_column_lifetime_stream(
            *right_recv[static_cast<std::size_t>(right_key)], runtime.join_stream);
        if (early_reclaim) {
            right_recv[static_cast<std::size_t>(right_key)].reset();
        }
    }
    record_event(materialize_stop, runtime.join_stream);
    materialize_phase.finish();
    if (materialize_range) materialize_range->finish();
    ALPS_CUDACHECK(cudaEventRecord(final_event, runtime.join_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(caller_stream, final_event, 0));
    // Returned buffers must not retain the private join stream as their
    // lifetime stream: callers may keep a result until after the lazy runtime
    // is destroyed. The caller-stream wait above makes this ownership transfer
    // safe without a host synchronization.
    for (auto& column : output) {
        set_column_lifetime_stream(*column, caller_stream);
    }
    record_event(total_stop, caller_stream);

    if (timing) {
        ALPS_CUDACHECK(cudaEventSynchronize(total_stop.value));
        timing->total_ms = elapsed(total_start.value, total_stop.value);
        timing->key_prepare_ms = elapsed(
            key_prepare_start.value, key_prepare_stop.value);
        timing->payload_prepare_ms = elapsed(
            payload_prepare_start.value, payload_prepare_stop.value);
        timing->key_exchange_ms = elapsed(
            key_exchange_start.value, key_exchange_stop.value);
        timing->payload_exchange_ms = elapsed(
            payload_exchange_start.value, payload_exchange_stop.value);
        timing->match_ms = elapsed(match_start.value, match_stop.value);
        timing->materialize_ms = elapsed(
            materialize_start.value, materialize_stop.value);
    }

    return gamax::make_table(std::move(output));
}

} // namespace alps::operators
