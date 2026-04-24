#pragma once

#include <cuda_runtime_api.h>
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

private:
    struct Pending {
        cudaEvent_t start {nullptr};
        cudaEvent_t stop {nullptr};
        int step_index {-1};
        StepKind kind {};
        const char* kind_name {""};
    };
    std::vector<Pending> pending_;
    std::vector<Row> rows_;
    bool finalized_ {false};
};

} // namespace gamax::exec
