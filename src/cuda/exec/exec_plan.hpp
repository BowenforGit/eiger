#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "expr/expr_fast.hpp"
#include "ir/schema.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/join/join.hpp"
#include "operators/project/project.hpp"
#include "operators/select/predicate.hpp"
#include "operators/select/select.hpp"
#include "operators/sort/sort.hpp"
#include "phys/phys.hpp"
#include "types/data_type.hpp"

namespace gamax::exec {

// ExecPlan is the representation produced by `lower_to_exec_plan`
// from an `ir::PlanNode` DAG. It is the bridge between the IR
// (declarative, rule-rewritable, ColumnId-keyed) and physical
// execution (`phys::*` calls or generated C++). Every field that the
// emitter previously computed per-call (ColumnId→ordinal resolution,
// `ExprTree` / `Predicate` construction, join-key extraction, reshape
// ordinals) lives here as pre-built data. An Executor or a
// CodeGenerator walks an ExecPlan without any IR involvement.

// Value identity, SSA-style. Each Step declares exactly one output
// value; subsequent Steps consume previous values by id. `-1` is the
// unset sentinel matching the codebase's `gamax::size_type` convention.
using ValueId = gamax::size_type;
inline constexpr ValueId kUnsetValueId {-1};

// A Value is an intermediate or final result in the plan. Its
// `schema` is positional (ColumnIds were retired at the IR→exec
// boundary); `last_used_step` is filled in by the lowering pass and
// drives the Executor's intermediate-table freeing.
struct Value {
    ValueId id {kUnsetValueId};
    gamax::ir::Schema schema;
    gamax::size_type last_used_step {-1};  // index into ExecPlan::steps
};

// ── Per-kind step payloads ────────────────────────────────────────

// SourceTable emits a pinned view into caller-owned data. `columns`
// is the list of source-table column names (catalog-qualified) the
// resolver must surface, **in output order** — the resulting view's
// `k`-th column must be the named column. Downstream ordinals index
// this narrow schema, not the full catalog one. Column-pruning at
// the scan replaces what used to be a SourceTable(full-catalog) +
// Reorder(pick-subset) pair.
struct SourceTablePayload {
    std::string table_name;
    std::vector<std::string> columns;
};
struct ReorderPayload {
    std::vector<gamax::size_type> ordinals;
};
// A Project step emits one output column per item. Numeric/ColRef
// items are expressed as ExprTrees; items whose value is a BOOL8
// predicate (including ones derived from StringOp leaves such as
// LIKE/STARTSWITH) are expressed as Predicates and evaluated via
// select_to_bitmask. The variant preserves per-item kind; output
// ordering matches the `items` vector. `ProjectItem` is declared
// once in `phys/phys.hpp`; re-used here.
// ── Per-payload hint fields ───────────────────────────────────────
//
// Each payload that has a runtime implementation choice carries a
// typed `hint` field. Default-constructed values match the engine's
// historical fixed-implementation behavior, so existing call sites
// produce identical execution. The lowering pass populates the field
// from `LowerOptions` overrides; the executor forwards
// `payload.hint` to the corresponding phys op. The bench harness
// drives operator-hint sweeps by setting `LowerOptions` from NVBench
// axes.
struct ProjectPayload {
    std::vector<gamax::phys::ProjectItem> items;
    gamax::project::project_options hint {};
};
struct FilterPayload {
    gamax::select::Predicate predicate;
    // Subset of input ordinals to materialize (in that order). Empty
    // means "all input columns" (the default, producing Filter's input
    // schema unchanged). Set by the lowering-time column-pruning rule
    // when some input column is touched only by the predicate and not
    // consumed by any downstream operator.
    std::vector<gamax::size_type> output_ordinals;
    gamax::select::select_options hint {};
};
struct EqJoinPayload {
    std::vector<gamax::size_type> lkeys;
    std::vector<gamax::size_type> rkeys;
    gamax::join::join_hint hint {};
};
struct SemiJoinPayload {
    std::vector<gamax::size_type> lkeys;
    std::vector<gamax::size_type> rkeys;
    gamax::join::join_hint hint {};
};
// LEFT ANTI JOIN — output schema is identical to SemiJoin (L only); only
// the underlying kernel differs (keep L rows whose key has NO match on
// R). Keeping a distinct payload type rather than a flag on SemiJoin so
// `std::visit` dispatch at the executor cleanly picks the right phys op.
struct AntiJoinPayload {
    std::vector<gamax::size_type> lkeys;
    std::vector<gamax::size_type> rkeys;
    gamax::join::join_hint hint {};
};
struct CrossJoinPayload {};
struct ScalarAggPayload {
    std::vector<gamax::phys::AggSpec> specs;
};
struct GroupByPayload {
    // `n_keys` defines the split of the (materialized) input table
    // into a key view (`[0, n_keys)`) and a value view
    // (`[n_keys, end)`) that the Executor passes to `phys::group_by`.
    gamax::size_type n_keys;
    std::vector<gamax::phys::AggSpec> specs;
    gamax::groupby::groupby_hint hint {};
};
// Distinct: column ordinals of the input Value that form the
// distinct-keys set. The executor selects those columns and passes
// them to `phys::distinct`; the selection is sparse, so the caller
// never materializes a dense keys-only intermediate upstream.
//
// `hint` mirrors `GroupByPayload::hint` — Distinct is algorithmically a
// no-agg groupby, so the same algorithm-selection and key-fusion knobs
// apply. Default-constructed (HASH, no SKF) takes the dedicated
// `hash::distinct` fast path; non-default routes through
// `gamax::groupby::groupby` to honor the algorithm and SKF flags.
struct DistinctPayload {
    std::vector<gamax::size_type> key_ordinals;
    gamax::groupby::groupby_hint hint {};
};
// Concat (SQL UNION ALL) has no payload — the N input ValueIds live
// on `Step::inputs` and the phys op walks them in order. The output
// schema is implied by the step's output Value.
struct ConcatPayload {};
struct SortPayload {
    std::vector<gamax::phys::SortSpec> keys;
    gamax::sort::sort_table_options hint {};
};
struct TopKPayload {
    std::vector<gamax::phys::SortSpec> keys;
    gamax::size_type k;
    gamax::sort::sort_table_options hint {};
};

using StepPayload = std::variant<
    SourceTablePayload,
    ReorderPayload,
    ProjectPayload,
    FilterPayload,
    EqJoinPayload,
    SemiJoinPayload,
    AntiJoinPayload,
    CrossJoinPayload,
    ScalarAggPayload,
    GroupByPayload,
    DistinctPayload,
    ConcatPayload,
    SortPayload,
    TopKPayload>;

// Duplicated across the variant as a kind tag for dispatch-by-switch
// and for printing, without needing std::visit everywhere.
enum class StepKind {
    SourceTable, Reorder, Project, Filter,
    EqJoin, SemiJoin, AntiJoin, CrossJoin,
    ScalarAgg, GroupBy, Distinct, Concat,
    Sort, TopK,
};

struct Step {
    StepKind kind;
    std::vector<ValueId> inputs;
    ValueId output;
    StepPayload payload;
};

struct ExecPlan {
    std::vector<Value> values;
    std::vector<Step>  steps;  // ordered; last step's output is the result
    std::vector<std::string> required_sources;
};

} // namespace gamax::exec
