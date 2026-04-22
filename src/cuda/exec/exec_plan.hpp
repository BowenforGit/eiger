#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "expr/expr_fast.hpp"
#include "ir/schema.hpp"
#include "operators/select/predicate.hpp"
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

struct SourceTablePayload {
    std::string table_name;
};
struct ReorderPayload {
    std::vector<gamax::size_type> ordinals;
};
struct ProjectPayload {
    std::vector<gamax::expression::ExprTree> exprs;
};
struct FilterPayload {
    gamax::select::Predicate predicate;
};
struct EqJoinPayload {
    std::vector<gamax::size_type> lkeys;
    std::vector<gamax::size_type> rkeys;
};
struct SemiJoinPayload {
    std::vector<gamax::size_type> lkeys;
    std::vector<gamax::size_type> rkeys;
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
};
struct SortPayload {
    std::vector<gamax::phys::SortSpec> keys;
};
struct TopKPayload {
    std::vector<gamax::phys::SortSpec> keys;
    gamax::size_type k;
};

using StepPayload = std::variant<
    SourceTablePayload,
    ReorderPayload,
    ProjectPayload,
    FilterPayload,
    EqJoinPayload,
    SemiJoinPayload,
    CrossJoinPayload,
    ScalarAggPayload,
    GroupByPayload,
    SortPayload,
    TopKPayload>;

// Duplicated across the variant as a kind tag for dispatch-by-switch
// and for printing, without needing std::visit everywhere.
enum class StepKind {
    SourceTable, Reorder, Project, Filter,
    EqJoin, SemiJoin, CrossJoin,
    ScalarAgg, GroupBy,
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
