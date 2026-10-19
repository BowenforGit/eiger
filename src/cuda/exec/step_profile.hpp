// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>
#include <map>
#include <ostream>
#include <utility>
#include <string>
#include <vector>

#include "exec/exec_plan.hpp"

namespace gamax::exec {

enum class PhaseCategory {
    Compute,
    Communication,
    Overhead,
};

enum class CommunicationPattern {
    None,
    Shuffle,
    Broadcast,
    Gather,
    Reduce,
    Sampling,
    Other,
};

// Exact user-payload accounting at the NCCL API boundary. These records do
// not include NCCL protocol headers or any lower-level fabric traffic.
enum class MessageApi {
    NcclSend,
    NcclRecv,
    NcclBroadcast,
    NcclAllReduce,
};

enum class MessageComponent {
    FixedValue,
    StringLength,
    StringChars,
    ScalarState,
};

enum class MessageSegment {
    Whole,
    Main,
    Pre,
    Post,
};

const char* message_api_name(MessageApi api) noexcept;
const char* message_direction_name(MessageApi api) noexcept;
const char* message_component_name(MessageComponent component) noexcept;
const char* message_segment_name(MessageSegment segment) noexcept;

struct MessageTraceDescriptor {
    std::int64_t column_ordinal {-1};
    gamax::data_type logical_type {};
    MessageComponent component {MessageComponent::FixedValue};
    MessageSegment segment {MessageSegment::Whole};
};

const char* phase_category_name(PhaseCategory category);
const char* communication_pattern_name(CommunicationPattern pattern);
std::int64_t table_data_bytes(gamax::table_view table) noexcept;
float phase_interval_union_ms(
    std::vector<std::pair<float, float>> intervals);

struct PhaseMetadata {
    std::string name;
    PhaseCategory category {PhaseCategory::Compute};
    CommunicationPattern communication_pattern {CommunicationPattern::None};
    std::string collective_primitive;
    std::int64_t bytes_sent {-1};
    std::int64_t bytes_received {-1};
    std::int64_t rows_sent {-1};
    std::int64_t rows_received {-1};
    int peer_count {-1};
    std::int64_t alignment_main_rows {-1};
    std::int64_t alignment_pre_rows {-1};
    std::int64_t alignment_post_rows {-1};
    std::int64_t alignment_main_bytes {-1};
    std::int64_t alignment_pre_bytes {-1};
    std::int64_t alignment_post_bytes {-1};
    // Optional identity for a phase that operates on exactly one logical
    // column. Empty / -1 means the phase is not column-specific or covers
    // multiple columns. Binary operators use left/right plus the original
    // input ordinal; staged unary operators may use schedule-specific sides
    // such as key/state plus the staged-table ordinal.
    std::string column_side;
    std::int64_t column_ordinal {-1};
};

// Per-step GPU timing captured via `cudaEvent` pairs on the executor's
// stream.
//
// Why events, not CPU `chrono`: phys ops launch kernels asynchronously
// on one stream and the executor only syncs at the end of `execute()`.
// Wrapping each dispatch in a `steady_clock` range would measure launch
// queueing (microseconds), not kernel runtime. `cudaEventRecord` on the
// same stream captures a timestamp at kernel-completion semantics, so
// the elapsed time between pre- and post-step events is real GPU work.
// The events are resolved in `finalize()`, which syncs once.
//
// Cost when disabled: zero. The sink is opt-in via a pointer on
// `phys::Context`; when null, the executor skips all event work.
//
// Thread-safety: one sink per `execute()` call. Not thread-safe.
class StepProfileSink {
public:
    struct Row {
        int step_index {-1};
        StepKind kind {};
        const char* kind_name {""};  // static storage (from step_kind_name)
        std::string label;
        std::string requested_implementation;
        std::string used_implementation;
        std::string fallback_reason;
        float gpu_ms {0.0f};
        float phase_union_ms {0.0f};
    };

    struct PhaseRow {
        int step_index {-1};
        StepKind step_kind {};
        const char* step_kind_name {""};
        std::string step_label;
        int phase_index {-1};
        PhaseMetadata metadata;
        float gpu_ms {0.0f};
        float start_ms {0.0f};
        float end_ms {0.0f};
    };

    struct MessageRow {
        int iteration {0};
        std::int64_t call_index {-1};
        int step_index {-1};
        StepKind step_kind {};
        const char* step_kind_name {""};
        std::string step_label;
        int phase_index {-1};
        std::string phase_name;
        CommunicationPattern communication_pattern {
            CommunicationPattern::None};
        std::string collective_primitive;
        MessageApi api {MessageApi::NcclSend};
        int peer {-1};
        int root {-1};
        std::int64_t column_ordinal {-1};
        gamax::data_type logical_type {};
        gamax::data_type element_type {};
        MessageComponent component {MessageComponent::FixedValue};
        MessageSegment segment {MessageSegment::Whole};
        std::int64_t element_count {0};
        std::int64_t element_size {0};
        std::int64_t payload_bytes {0};
    };

    using PhaseHandle = std::uint64_t;
    static constexpr PhaseHandle invalid_phase_handle = 0;

    explicit StepProfileSink(
        bool collect_phases = false,
        bool collect_messages = false,
        int message_iteration = 0)
        : collect_phases_(collect_phases),
          collect_messages_(collect_messages),
          message_iteration_(message_iteration) {}
    ~StepProfileSink();
    StepProfileSink(const StepProfileSink&) = delete;
    StepProfileSink& operator=(const StepProfileSink&) = delete;

    // Called by the executor immediately before / after dispatching a
    // step. Each `begin_step` must be followed by exactly one
    // `end_step` on the same stream.
    void begin_step(
        int step_index,
        StepKind kind,
        std::string label,
        cudaStream_t stream);
    void end_step(cudaStream_t stream);

    // Attach a machine-readable implementation decision to the active step.
    // Empty fallback_reason means the requested implementation was used.
    void set_active_step_implementation(
        std::string requested,
        std::string used,
        std::string fallback_reason = {});

    bool collects_phases() const noexcept { return collect_phases_; }
    bool collects_messages() const noexcept { return collect_messages_; }
    PhaseHandle begin_phase(PhaseMetadata metadata, cudaStream_t stream);
    void update_phase_transfer(
        PhaseHandle handle,
        std::int64_t bytes_sent,
        std::int64_t bytes_received,
        std::int64_t rows_sent,
        std::int64_t rows_received,
        int peer_count);
    void update_phase_alignment(
        PhaseHandle handle,
        std::int64_t main_rows,
        std::int64_t pre_rows,
        std::int64_t post_rows,
        std::int64_t main_bytes,
        std::int64_t pre_bytes,
        std::int64_t post_bytes);
    void end_phase(PhaseHandle handle, cudaStream_t stream);

    void record_message(
        MessageApi api,
        int peer,
        int root,
        gamax::data_type element_type,
        std::int64_t element_count,
        std::int64_t element_size,
        MessageTraceDescriptor descriptor = {});

    // Sync the stream and resolve all recorded event pairs into rows.
    // Idempotent.
    void finalize(cudaStream_t stream);

    const std::vector<Row>& rows() const { return rows_; }
    const std::vector<PhaseRow>& phase_rows() const { return phase_rows_; }
    const std::vector<MessageRow>& message_rows() const { return message_rows_; }

    // Dump rows as CSV with a leading `query` column identifying this
    // run. `write_header=true` emits the header once (typically on the
    // first append to a new file).
    void to_csv(std::ostream& os, const std::string& query_label,
                bool write_header) const;

    // ─────────────────────────────────────────────────────────────
    // Aggregating mode — for the NVBench bench harness.
    //
    // Each NVBench sample re-records the per-step event pairs. We
    // call `ingest_sample(stream)` after every sample to drain those
    // events into running per-step statistics (count, sum, sum of
    // squares, min, max). At end of cell, `to_summary_csv` emits one
    // row per step_index with mean/min/max/stddev across all
    // samples. The schema is wider than `rows_`/`to_csv`'s
    // single-shot form — different consumer (figure breakdown)
    // wants different columns.
    //
    // The two modes (single-shot rows + aggregating summary) do not
    // mix per sink instance. Pick one when constructing the sink.
    // ─────────────────────────────────────────────────────────────
    struct Summary {
        int step_index {-1};
        StepKind kind {};
        const char* kind_name {""};
        std::string label;
        int64_t n_samples {0};
        double mean_ms {0.0};
        double min_ms  {0.0};
        double max_ms  {0.0};
        double stddev_ms {0.0};
    };

    // Sync the stream, resolve all currently-pending event pairs,
    // accumulate them into the per-step aggregator, then destroy the
    // events. Safe to call repeatedly. NVBench's `exec_tag::sync`
    // already syncs around timer.start/stop, so the sync here is
    // belt-and-suspenders cheap.
    void ingest_sample(cudaStream_t stream);

    // Sync and destroy the currently-pending event pairs without
    // accumulating them. Used to drop a warmup sample whose JIT /
    // first-touch costs would inflate the mean/stddev.
    void discard_pending(cudaStream_t stream);

    // Snapshot of the aggregator. One entry per distinct step_index.
    // Order: ascending by step_index (matches plan-natural order).
    std::vector<Summary> summary() const;

    // Dump the aggregated summary as CSV. Columns:
    //   query,scale,step_index,step_kind,n_samples,mean_ms,min_ms,max_ms,stddev_ms
    // `write_header=true` emits the header once.
    void to_summary_csv(std::ostream& os,
                        const std::string& query_label,
                        int64_t scale_factor,
                        bool write_header) const;

private:
    struct Pending {
        cudaEvent_t start {nullptr};
        cudaEvent_t stop {nullptr};
        int step_index {-1};
        StepKind kind {};
        const char* kind_name {""};
        std::string label;
        std::string requested_implementation;
        std::string used_implementation;
        std::string fallback_reason;
    };
    struct AggState {
        StepKind kind {};
        const char* kind_name {""};
        std::string label;
        int64_t n {0};
        double sum_ms {0.0};
        double sum_sq_ms {0.0};
        double min_ms {0.0};
        double max_ms {0.0};
        bool initialized {false};
    };
    struct PendingPhase {
        cudaEvent_t start {nullptr};
        cudaEvent_t stop {nullptr};
        cudaEvent_t step_origin {nullptr};
        int step_index {-1};
        StepKind step_kind {};
        const char* step_kind_name {""};
        std::string step_label;
        int phase_index {-1};
        PhaseMetadata metadata;
    };
    std::vector<Pending> pending_;
    std::vector<PendingPhase> pending_phases_;
    std::vector<Row> rows_;
    std::vector<PhaseRow> phase_rows_;
    std::vector<MessageRow> message_rows_;
    std::map<int, AggState> aggs_;  // ordered by step_index
    bool collect_phases_ {false};
    bool collect_messages_ {false};
    int message_iteration_ {0};
    std::int64_t next_message_call_index_ {0};
    bool step_active_ {false};
    std::map<PhaseHandle, std::size_t> active_phases_;
    int active_step_index_ {-1};
    StepKind active_step_kind_ {};
    const char* active_step_kind_name_ {""};
    std::string active_step_label_;
    int next_phase_index_ {0};
    PhaseHandle next_phase_handle_ {1};
    bool finalized_ {false};
};

// A phase nested under the currently active physical step. Multiple ranges may
// be active concurrently on different CUDA streams. NVTX is emitted even
// without a timing sink so NSYS profile mode sees the same phase hierarchy.
class PhaseRange {
public:
    PhaseRange(
        StepProfileSink* sink,
        cudaStream_t stream,
        PhaseMetadata metadata,
        bool emit_nvtx = false);
    PhaseRange(
        const gamax::phys::Context& context,
        PhaseMetadata metadata);
    ~PhaseRange() noexcept;
    PhaseRange(const PhaseRange&) = delete;
    PhaseRange& operator=(const PhaseRange&) = delete;

    void set_transfer(
        std::int64_t bytes_sent,
        std::int64_t bytes_received,
        std::int64_t rows_sent,
        std::int64_t rows_received,
        int peer_count);
    void set_alignment(
        std::int64_t main_rows,
        std::int64_t pre_rows,
        std::int64_t post_rows,
        std::int64_t main_bytes,
        std::int64_t pre_bytes,
        std::int64_t post_bytes);

private:
    StepProfileSink* sink_ {nullptr};
    StepProfileSink::PhaseHandle handle_ {
        StepProfileSink::invalid_phase_handle};
    cudaStream_t stream_ {cudaStreamDefault};
    bool nvtx_active_ {false};
};

} // namespace gamax::exec
