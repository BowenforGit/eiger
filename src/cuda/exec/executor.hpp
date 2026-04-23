#pragma once

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

} // namespace gamax::exec
