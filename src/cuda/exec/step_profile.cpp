// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "exec/step_profile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nvtx3/nvToolsExt.h>

#include "exec/print.hpp"

namespace gamax::exec {

const char* message_api_name(MessageApi api) noexcept {
    switch (api) {
        case MessageApi::NcclSend: return "ncclSend";
        case MessageApi::NcclRecv: return "ncclRecv";
        case MessageApi::NcclBroadcast: return "ncclBroadcast";
        case MessageApi::NcclAllReduce: return "ncclAllReduce";
    }
    return "unknown";
}

const char* message_direction_name(MessageApi api) noexcept {
    switch (api) {
        case MessageApi::NcclSend: return "send";
        case MessageApi::NcclRecv: return "receive";
        case MessageApi::NcclBroadcast:
        case MessageApi::NcclAllReduce: return "collective";
    }
    return "unknown";
}

const char* message_component_name(MessageComponent component) noexcept {
    switch (component) {
        case MessageComponent::FixedValue: return "fixed_value";
        case MessageComponent::StringLength: return "string_length";
        case MessageComponent::StringChars: return "string_chars";
        case MessageComponent::ScalarState: return "scalar_state";
    }
    return "unknown";
}

const char* message_segment_name(MessageSegment segment) noexcept {
    switch (segment) {
        case MessageSegment::Whole: return "whole";
        case MessageSegment::Main: return "main";
        case MessageSegment::Pre: return "pre";
        case MessageSegment::Post: return "post";
    }
    return "unknown";
}

const char* phase_category_name(PhaseCategory category) {
    switch (category) {
        case PhaseCategory::Compute: return "compute";
        case PhaseCategory::Communication: return "communication";
        case PhaseCategory::Overhead: return "overhead";
    }
    return "unknown";
}

const char* communication_pattern_name(CommunicationPattern pattern) {
    switch (pattern) {
        case CommunicationPattern::None: return "none";
        case CommunicationPattern::Shuffle: return "shuffle";
        case CommunicationPattern::Broadcast: return "broadcast";
        case CommunicationPattern::Gather: return "gather";
        case CommunicationPattern::Reduce: return "reduce";
        case CommunicationPattern::Sampling: return "sampling";
        case CommunicationPattern::Other: return "other";
    }
    return "unknown";
}

std::int64_t table_data_bytes(gamax::table_view table) noexcept {
    std::int64_t total = 0;
    for (gamax::size_type index = 0; index < table.num_columns(); ++index) {
        const auto column = table.get_column(index);
        if (column.type().id() == gamax::type_id::STRING) {
            total += static_cast<std::int64_t>(column.auxiliary_bytes());
            total += static_cast<std::int64_t>(column.size() + 1) *
                static_cast<std::int64_t>(sizeof(gamax::size_type));
        } else {
            total += static_cast<std::int64_t>(column.size()) *
                static_cast<std::int64_t>(gamax::size_of(column.type()));
        }
    }
    return total;
}

float phase_interval_union_ms(
    std::vector<std::pair<float, float>> intervals)
{
    if (intervals.empty()) return 0.0f;
    for (auto& interval : intervals) {
        if (interval.second < interval.first) std::swap(interval.first, interval.second);
    }
    std::sort(intervals.begin(), intervals.end());
    float total = 0.0f;
    float start = intervals.front().first;
    float end = intervals.front().second;
    for (std::size_t index = 1; index < intervals.size(); ++index) {
        if (intervals[index].first <= end) {
            end = std::max(end, intervals[index].second);
        } else {
            total += end - start;
            start = intervals[index].first;
            end = intervals[index].second;
        }
    }
    return total + end - start;
}

StepProfileSink::~StepProfileSink() {
    // Events may linger if finalize() was never called (e.g. the
    // execute() call threw). Destroy whatever is left; cudaEventDestroy
    // is safe on recorded-but-unsynced events.
    for (auto& p : pending_) {
        if (p.start) cudaEventDestroy(p.start);
        if (p.stop) cudaEventDestroy(p.stop);
    }
    for (auto& p : pending_phases_) {
        if (p.start) cudaEventDestroy(p.start);
        if (p.stop) cudaEventDestroy(p.stop);
    }
}

void StepProfileSink::begin_step(
    int step_index,
    StepKind kind,
    std::string label,
    cudaStream_t stream) {
    if (step_active_) {
        throw std::runtime_error(
            "StepProfileSink only supports one active step");
    }
    Pending p;
    p.step_index = step_index;
    p.kind = kind;
    p.kind_name = step_kind_name(kind);
    p.label = std::move(label);
    cudaEventCreate(&p.start);
    cudaEventCreate(&p.stop);
    cudaEventRecord(p.start, stream);
    pending_.push_back(p);
    step_active_ = true;
    active_step_index_ = step_index;
    active_step_kind_ = kind;
    active_step_kind_name_ = p.kind_name;
    active_step_label_ = p.label;
    next_phase_index_ = 0;
}

void StepProfileSink::end_step(cudaStream_t stream) {
    if (!step_active_ || pending_.empty()) {
        throw std::runtime_error(
            "StepProfileSink::end_step called without matching begin_step");
    }
    if (!active_phases_.empty()) {
        throw std::runtime_error(
            "StepProfileSink::end_step called with an active phase");
    }
    cudaEventRecord(pending_.back().stop, stream);
    step_active_ = false;
    active_step_index_ = -1;
    active_step_kind_name_ = "";
    active_step_label_.clear();
}

void StepProfileSink::set_active_step_implementation(
    std::string requested,
    std::string used,
    std::string fallback_reason)
{
    if (!step_active_ || pending_.empty()) {
        throw std::runtime_error(
            "StepProfileSink::set_active_step_implementation called outside an active step");
    }
    auto& pending = pending_.back();
    pending.requested_implementation = std::move(requested);
    pending.used_implementation = std::move(used);
    pending.fallback_reason = std::move(fallback_reason);
}

StepProfileSink::PhaseHandle StepProfileSink::begin_phase(
    PhaseMetadata metadata, cudaStream_t stream)
{
    if (!collect_phases_) return invalid_phase_handle;
    if (!step_active_) {
        throw std::runtime_error(
            "StepProfileSink::begin_phase called outside an active step");
    }
    PendingPhase phase;
    const auto handle = next_phase_handle_++;
    phase.step_index = active_step_index_;
    phase.step_kind = active_step_kind_;
    phase.step_kind_name = active_step_kind_name_;
    phase.step_label = active_step_label_;
    phase.phase_index = next_phase_index_++;
    phase.metadata = std::move(metadata);
    phase.step_origin = pending_.back().start;
    cudaEventCreate(&phase.start);
    cudaEventCreate(&phase.stop);
    cudaEventRecord(phase.start, stream);
    pending_phases_.push_back(std::move(phase));
    active_phases_.emplace(handle, pending_phases_.size() - 1);
    return handle;
}

void StepProfileSink::update_phase_transfer(
    PhaseHandle handle,
    std::int64_t bytes_sent,
    std::int64_t bytes_received,
    std::int64_t rows_sent,
    std::int64_t rows_received,
    int peer_count)
{
    if (!collect_phases_) return;
    const auto active = active_phases_.find(handle);
    if (active == active_phases_.end()) {
        throw std::runtime_error(
            "StepProfileSink::update_phase_transfer has no active phase");
    }
    auto& metadata = pending_phases_[active->second].metadata;
    metadata.bytes_sent = bytes_sent;
    metadata.bytes_received = bytes_received;
    metadata.rows_sent = rows_sent;
    metadata.rows_received = rows_received;
    metadata.peer_count = peer_count;
}

void StepProfileSink::update_phase_alignment(
    PhaseHandle handle,
    std::int64_t main_rows,
    std::int64_t pre_rows,
    std::int64_t post_rows,
    std::int64_t main_bytes,
    std::int64_t pre_bytes,
    std::int64_t post_bytes)
{
    if (!collect_phases_) return;
    const auto active = active_phases_.find(handle);
    if (active == active_phases_.end()) {
        throw std::runtime_error(
            "StepProfileSink::update_phase_alignment has no active phase");
    }
    auto& metadata = pending_phases_[active->second].metadata;
    metadata.alignment_main_rows = main_rows;
    metadata.alignment_pre_rows = pre_rows;
    metadata.alignment_post_rows = post_rows;
    metadata.alignment_main_bytes = main_bytes;
    metadata.alignment_pre_bytes = pre_bytes;
    metadata.alignment_post_bytes = post_bytes;
}

void StepProfileSink::end_phase(PhaseHandle handle, cudaStream_t stream) {
    if (!collect_phases_) return;
    const auto active = active_phases_.find(handle);
    if (active == active_phases_.end()) {
        throw std::runtime_error(
            "StepProfileSink::end_phase called without matching begin_phase");
    }
    cudaEventRecord(pending_phases_[active->second].stop, stream);
    active_phases_.erase(active);
}

void StepProfileSink::record_message(
    MessageApi api,
    int peer,
    int root,
    gamax::data_type element_type,
    std::int64_t element_count,
    std::int64_t element_size,
    MessageTraceDescriptor descriptor)
{
    if (!collect_messages_) return;
    if (!step_active_) {
        throw std::runtime_error(
            "StepProfileSink::record_message called outside an active step");
    }
    if (active_phases_.empty()) {
        throw std::runtime_error(
            "StepProfileSink::record_message called outside an active phase");
    }
    if (element_count < 0 || element_size < 0 ||
        (element_size != 0 &&
         element_count > std::numeric_limits<std::int64_t>::max() /
             element_size)) {
        throw std::overflow_error("NCCL message payload size overflows int64");
    }

    // Legacy execution is serialized. Choosing the most recently opened phase
    // also gives deterministic attribution if an outer range remains active.
    const auto active = active_phases_.rbegin();
    const auto& phase = pending_phases_[active->second];
    if (descriptor.logical_type.id() == gamax::type_id::EMPTY) {
        descriptor.logical_type = element_type;
    }
    message_rows_.push_back(MessageRow{
        message_iteration_, next_message_call_index_++,
        active_step_index_, active_step_kind_, active_step_kind_name_,
        active_step_label_, phase.phase_index, phase.metadata.name,
        phase.metadata.communication_pattern,
        phase.metadata.collective_primitive,
        api, peer, root, descriptor.column_ordinal,
        descriptor.logical_type, element_type, descriptor.component,
        descriptor.segment, element_count, element_size,
        element_count * element_size});
}

void StepProfileSink::finalize(cudaStream_t stream) {
    if (finalized_) return;
    if (step_active_ || !active_phases_.empty()) {
        throw std::runtime_error(
            "StepProfileSink::finalize called with an active range");
    }
    cudaStreamSynchronize(stream);
    for (auto& phase : pending_phases_) {
        cudaEventSynchronize(phase.stop);
    }
    phase_rows_.reserve(pending_phases_.size());
    for (auto& phase : pending_phases_) {
        float start_ms = 0.0f;
        float end_ms = 0.0f;
        cudaEventElapsedTime(&start_ms, phase.step_origin, phase.start);
        cudaEventElapsedTime(&end_ms, phase.step_origin, phase.stop);
        phase_rows_.push_back(PhaseRow{
            phase.step_index,
            phase.step_kind,
            phase.step_kind_name,
            phase.step_label,
            phase.phase_index,
            std::move(phase.metadata),
            end_ms - start_ms,
            start_ms,
            end_ms});
        cudaEventDestroy(phase.start);
        cudaEventDestroy(phase.stop);
        phase.start = nullptr;
        phase.stop = nullptr;
    }
    pending_phases_.clear();
    rows_.reserve(pending_.size());
    for (auto& p : pending_) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, p.start, p.stop);
        std::vector<std::pair<float, float>> intervals;
        for (const auto& phase : phase_rows_) {
            if (phase.step_index == p.step_index) {
                const auto start = std::max(0.0f, phase.start_ms);
                const auto end = std::min(ms, phase.end_ms);
                if (start < end) intervals.emplace_back(start, end);
            }
        }
        rows_.push_back(Row{
            p.step_index, p.kind, p.kind_name, p.label,
            std::move(p.requested_implementation),
            std::move(p.used_implementation),
            std::move(p.fallback_reason), ms,
            phase_interval_union_ms(std::move(intervals))});
        cudaEventDestroy(p.start);
        cudaEventDestroy(p.stop);
        p.start = nullptr;
        p.stop = nullptr;
    }
    pending_.clear();
    finalized_ = true;
}

void StepProfileSink::to_csv(std::ostream& os,
                             const std::string& query_label,
                             bool write_header) const {
    if (write_header) {
        os << "query,step_index,step_kind,step_label,"
              "requested_implementation,used_implementation,"
              "fallback_reason,gpu_ms\n";
    }
    for (const auto& r : rows_) {
        os << query_label << "," << r.step_index << ","
           << r.kind_name << ",\"" << r.label << "\",\""
           << r.requested_implementation << "\",\""
           << r.used_implementation << "\",\""
           << r.fallback_reason << "\"," << r.gpu_ms << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────
// Aggregating mode
// ─────────────────────────────────────────────────────────────────────

void StepProfileSink::discard_pending(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    // Phase events may have been recorded on worker streams other than the
    // caller stream.  Drain their stop events before releasing them so this
    // cleanup path has the same cross-stream lifetime guarantee as finalize().
    for (auto& phase : pending_phases_) {
        if (phase.stop) cudaEventSynchronize(phase.stop);
    }
    for (auto& p : pending_) {
        if (p.start) cudaEventDestroy(p.start);
        if (p.stop)  cudaEventDestroy(p.stop);
        p.start = nullptr;
        p.stop  = nullptr;
    }
    pending_.clear();
    for (auto& phase : pending_phases_) {
        if (phase.start) cudaEventDestroy(phase.start);
        if (phase.stop) cudaEventDestroy(phase.stop);
        phase.start = nullptr;
        phase.stop = nullptr;
    }
    pending_phases_.clear();
    step_active_ = false;
    active_phases_.clear();
}

void StepProfileSink::ingest_sample(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    // Aggregating mode currently discards detailed phase rows, but phases can
    // still originate on auxiliary streams.  Their events must be complete
    // before destruction just as they are in finalize().
    for (auto& phase : pending_phases_) {
        if (phase.stop) cudaEventSynchronize(phase.stop);
    }
    for (auto& p : pending_) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, p.start, p.stop);

        auto& a = aggs_[p.step_index];
        if (!a.initialized) {
            a.kind = p.kind;
            a.kind_name = p.kind_name;
            a.label = p.label;
            a.min_ms = ms;
            a.max_ms = ms;
            a.initialized = true;
        } else {
            a.min_ms = std::min(a.min_ms, static_cast<double>(ms));
            a.max_ms = std::max(a.max_ms, static_cast<double>(ms));
        }
        a.n++;
        a.sum_ms    += ms;
        a.sum_sq_ms += static_cast<double>(ms) * static_cast<double>(ms);

        cudaEventDestroy(p.start);
        cudaEventDestroy(p.stop);
        p.start = nullptr;
        p.stop  = nullptr;
    }
    pending_.clear();
    for (auto& phase : pending_phases_) {
        if (phase.start) cudaEventDestroy(phase.start);
        if (phase.stop) cudaEventDestroy(phase.stop);
        phase.start = nullptr;
        phase.stop = nullptr;
    }
    pending_phases_.clear();
    step_active_ = false;
    active_phases_.clear();
}

std::vector<StepProfileSink::Summary>
StepProfileSink::summary() const {
    std::vector<Summary> out;
    out.reserve(aggs_.size());
    for (const auto& kv : aggs_) {
        const auto& a = kv.second;
        Summary s;
        s.step_index = kv.first;
        s.kind = a.kind;
        s.kind_name = a.kind_name;
        s.label = a.label;
        s.n_samples = a.n;
        if (a.n > 0) {
            const double mean = a.sum_ms / static_cast<double>(a.n);
            s.mean_ms = mean;
            // Population variance: E[x^2] - E[x]^2. Clamp negative
            // values from float rounding to 0 before sqrt.
            const double var =
                (a.sum_sq_ms / static_cast<double>(a.n)) - mean * mean;
            s.stddev_ms = (var > 0.0) ? std::sqrt(var) : 0.0;
            s.min_ms = a.min_ms;
            s.max_ms = a.max_ms;
        }
        out.push_back(s);
    }
    return out;
}

void StepProfileSink::to_summary_csv(std::ostream& os,
                                     const std::string& query_label,
                                     int64_t scale_factor,
                                     bool write_header) const {
    if (write_header) {
        os << "query,scale,step_index,step_kind,n_samples,"
              "mean_ms,min_ms,max_ms,stddev_ms\n";
    }
    for (const auto& s : summary()) {
        os << query_label << "," << scale_factor << ","
           << s.step_index << "," << s.kind_name << ","
           << s.n_samples << ","
           << s.mean_ms << "," << s.min_ms << ","
           << s.max_ms  << "," << s.stddev_ms << "\n";
    }
}

PhaseRange::PhaseRange(
    StepProfileSink* sink,
    cudaStream_t stream,
    PhaseMetadata metadata,
    bool emit_nvtx)
    : sink_(sink && sink->collects_phases() ? sink : nullptr), stream_(stream)
{
    if (sink_ || emit_nvtx) {
        std::string label = "phase:";
        label += phase_category_name(metadata.category);
        if (metadata.communication_pattern != CommunicationPattern::None) {
            label += ":";
            label += communication_pattern_name(metadata.communication_pattern);
        }
        label += ":";
        label += metadata.name;
        nvtxRangePushA(label.c_str());
        nvtx_active_ = true;
    }
    try {
        if (sink_) handle_ = sink_->begin_phase(std::move(metadata), stream_);
    } catch (...) {
        if (nvtx_active_) nvtxRangePop();
        throw;
    }
}

PhaseRange::PhaseRange(
    const gamax::phys::Context& context,
    PhaseMetadata metadata)
    : PhaseRange(
        context.profile_sink, context.stream, std::move(metadata),
        context.emit_phase_nvtx)
{
}

PhaseRange::~PhaseRange() noexcept {
    try {
        if (sink_) sink_->end_phase(handle_, stream_);
    } catch (...) {
        // Destructors must not mask an operator exception. Contract misuse is
        // still rejected by begin_phase/end_step/finalize on normal paths.
    }
    if (nvtx_active_) nvtxRangePop();
}

void PhaseRange::set_transfer(
    std::int64_t bytes_sent,
    std::int64_t bytes_received,
    std::int64_t rows_sent,
    std::int64_t rows_received,
    int peer_count)
{
    if (sink_) {
        sink_->update_phase_transfer(
            handle_,
            bytes_sent, bytes_received,
            rows_sent, rows_received, peer_count);
    }
}

void PhaseRange::set_alignment(
    std::int64_t main_rows,
    std::int64_t pre_rows,
    std::int64_t post_rows,
    std::int64_t main_bytes,
    std::int64_t pre_bytes,
    std::int64_t post_bytes)
{
    if (sink_) {
        sink_->update_phase_alignment(
            handle_, main_rows, pre_rows, post_rows,
            main_bytes, pre_bytes, post_bytes);
    }
}

} // namespace gamax::exec
