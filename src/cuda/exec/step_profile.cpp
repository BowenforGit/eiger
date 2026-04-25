#include "exec/step_profile.hpp"

#include <algorithm>
#include <cmath>
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

// ─────────────────────────────────────────────────────────────────────
// Aggregating mode
// ─────────────────────────────────────────────────────────────────────

void StepProfileSink::discard_pending(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    for (auto& p : pending_) {
        if (p.start) cudaEventDestroy(p.start);
        if (p.stop)  cudaEventDestroy(p.stop);
        p.start = nullptr;
        p.stop  = nullptr;
    }
    pending_.clear();
}

void StepProfileSink::ingest_sample(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    for (auto& p : pending_) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, p.start, p.stop);

        auto& a = aggs_[p.step_index];
        if (!a.initialized) {
            a.kind = p.kind;
            a.kind_name = p.kind_name;
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

} // namespace gamax::exec
