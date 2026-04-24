#include "exec/step_profile.hpp"

#include <stdexcept>

#include "exec/print.hpp"

namespace gamax::exec {

StepProfileSink::~StepProfileSink() {
    // Events may linger if finalize() was never called (e.g. the
    // execute() call threw). Destroy whatever is left; cudaEventDestroy
    // is safe on recorded-but-unsynced events.
    for (auto& p : pending_) {
        if (p.start) cudaEventDestroy(p.start);
        if (p.stop) cudaEventDestroy(p.stop);
    }
}

void StepProfileSink::begin_step(int step_index, StepKind kind,
                                 cudaStream_t stream) {
    Pending p;
    p.step_index = step_index;
    p.kind = kind;
    p.kind_name = step_kind_name(kind);
    cudaEventCreate(&p.start);
    cudaEventCreate(&p.stop);
    cudaEventRecord(p.start, stream);
    pending_.push_back(p);
}

void StepProfileSink::end_step(cudaStream_t stream) {
    if (pending_.empty()) {
        throw std::runtime_error(
            "StepProfileSink::end_step called without matching begin_step");
    }
    cudaEventRecord(pending_.back().stop, stream);
}

void StepProfileSink::finalize(cudaStream_t stream) {
    if (finalized_) return;
    cudaStreamSynchronize(stream);
    rows_.reserve(pending_.size());
    for (auto& p : pending_) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, p.start, p.stop);
        rows_.push_back(Row{p.step_index, p.kind, p.kind_name, ms});
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
        os << "query,step_index,step_kind,gpu_ms\n";
    }
    for (const auto& r : rows_) {
        os << query_label << "," << r.step_index << ","
           << r.kind_name << "," << r.gpu_ms << "\n";
    }
}

} // namespace gamax::exec
