#pragma once

#include <cuda_runtime_api.h>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "exec/exec_plan.hpp"

namespace gamax::exec {

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
        float gpu_ms {0.0f};
    };

    StepProfileSink() = default;
    ~StepProfileSink();
    StepProfileSink(const StepProfileSink&) = delete;
    StepProfileSink& operator=(const StepProfileSink&) = delete;

    // Called by the executor immediately before / after dispatching a
    // step. Each `begin_step` must be followed by exactly one
    // `end_step` on the same stream.
    void begin_step(int step_index, StepKind kind, cudaStream_t stream);
    void end_step(cudaStream_t stream);

    // Sync the stream and resolve all recorded event pairs into rows.
    // Idempotent.
    void finalize(cudaStream_t stream);

    const std::vector<Row>& rows() const { return rows_; }

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
    };
    struct AggState {
        StepKind kind {};
        const char* kind_name {""};
        int64_t n {0};
        double sum_ms {0.0};
        double sum_sq_ms {0.0};
        double min_ms {0.0};
        double max_ms {0.0};
        bool initialized {false};
    };
    std::vector<Pending> pending_;
    std::vector<Row> rows_;
    std::map<int, AggState> aggs_;  // ordered by step_index
    bool finalized_ {false};
};

} // namespace gamax::exec
