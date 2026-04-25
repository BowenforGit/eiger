#pragma once

#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <string>

#include "exec/exec_plan.hpp"
#include "memory/table.hpp"
#include "phys/phys.hpp"

namespace gamax::exec {

// Source-table resolver: given a table name and the requested column
// list from a SourceTable step, returns a non-owning `table_view` into
// caller-owned data. The view's `k`-th column must be the column named
// `columns[k]` — the Executor relies on this positional contract (no
// name lookup happens after this call). The returned view must outlive
// the `execute()` call.
using SourceResolver = std::function<gamax::table_view(
    const std::string& name,
    const std::vector<std::string>& columns)>;

// Walk an ExecPlan step-by-step, dispatching to `phys::*` with the
// pre-built payloads. Returns the final step's output table.
//
// The Executor maintains owned `unique_ptr<gamax::table>` per
// intermediate Value and drops each value after its
// `last_used_step` — the same lifetime analysis the lowering pass
// computed. A future version can run across multiple streams; v1 is
// single-stream as captured on the `phys::Context`.
std::unique_ptr<gamax::table> execute(
    const ExecPlan& plan,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx);

// ─────────────────────────────────────────────────────────────────────
// Two-phase execution path for benchmarking.
//
// The point: NVBench's wall-time measurement should not include the
// SourceTable steps' H2D/decode work. Splitting `execute()` lets the
// caller do all SourceTable resolution outside NVBench's timed region
// (`exec_tag::timer`), and the timed region runs only the compute.
//
//   Phase 1: bench_resolve_sources(plan, resolver, stream)
//     Walks the plan, evaluates every SourceTable step, syncs the
//     stream so all H2Ds are complete on return. Returns an opaque
//     `BenchExecState` carrying the populated `live` map.
//
//   Phase 2: bench_run_steps(plan, std::move(state), ctx)
//     Walks the plan again, skips SourceTable steps, dispatches the
//     rest. Returns the final step's output table. Honors the same
//     `last_used_step` freeing as `execute()`.
//
// Phase 1 is intended to run untimed. Phase 2 is what NVBench should
// time. The two phases share the per-step NVTX + profile-sink
// instrumentation with `execute()` via the `step_dispatch` helper.
// ─────────────────────────────────────────────────────────────────────
class BenchExecState {
public:
    BenchExecState();
    BenchExecState(BenchExecState&&) noexcept;
    BenchExecState& operator=(BenchExecState&&) noexcept;
    BenchExecState(const BenchExecState&) = delete;
    BenchExecState& operator=(const BenchExecState&) = delete;
    ~BenchExecState();

    struct Impl;
    Impl* impl() { return impl_.get(); }
    const Impl* impl() const { return impl_.get(); }

private:
    std::unique_ptr<Impl> impl_;
};

BenchExecState bench_resolve_sources(
    const ExecPlan& plan,
    const SourceResolver& source_resolver,
    cudaStream_t stream);

std::unique_ptr<gamax::table> bench_run_steps(
    const ExecPlan& plan,
    BenchExecState state,
    const gamax::phys::Context& ctx);

} // namespace gamax::exec
