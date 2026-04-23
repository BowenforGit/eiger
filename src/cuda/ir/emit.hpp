#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "ir/catalog.hpp"
#include "ir/plan_node.hpp"
#include "memory/table.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/project/project.hpp"
#include "operators/select/select.hpp"

namespace gamax::ir {

// Context threaded through the emit pass.
//
// - `stream`: CUDA stream for all physical ops.
// - `catalog`: used by Scan to map column names to the ordinal positions
//   in source tables (both the IR and the catalog agree on the declared
//   column order).
// - `resolve_table`: callback returning the source table_view for a
//   given table name and the list of columns the Scan requested (in
//   output order). The returned view's `k`-th column must correspond
//   to `columns[k]` and must outlive the emit call.
// - Tuning hints for each operator follow the physical operators'
//   existing option structs; defaults are fine for a first-cut run.
struct EmitContext {
    cudaStream_t stream {cudaStreamDefault};
    const Catalog* catalog {nullptr};
    std::function<gamax::table_view(const std::string& name,
                                    const std::vector<std::string>& cols)>
        resolve_table;

    gamax::select::select_options   select_opts;
    gamax::project::project_options project_opts;
    gamax::groupby::groupby_hint    groupby_hint;
};

// Walk a PlanNode DAG and produce the final result table.
//
// Supported in this revision:
//   Scan, Filter (numeric / compound-numeric predicates),
//   Project, Aggregate (scalar and grouped; SUM / MIN / MAX / AVG),
//   Sort.
//
// Unsupported NodeKinds and AggKinds (Limit, Join variants, Distinct,
// Count / CountAll / CountDistinct, StringOp filters) throw
// std::runtime_error at emit time with a clear "not yet implemented"
// message — the IR is intentionally a richer logical surface than the
// current physical backend.
std::unique_ptr<gamax::table> emit(const PlanPtr& root, const EmitContext& ctx);

} // namespace gamax::ir
