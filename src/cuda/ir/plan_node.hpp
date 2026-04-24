#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ir/catalog.hpp"
#include "ir/column_id.hpp"
#include "ir/expr.hpp"
#include "ir/schema.hpp"
#include "types/data_type.hpp"  // for gamax::sort_order, gamax::size_type

namespace gamax::ir {

// Operator kinds needed to express the full TPC-H workload. Future
// additions (Window, Union, Values, ...) are forward-compatible: a new
// kind never invalidates existing plans.
//
// The IR is intentionally richer than the GPU backend today. A logical
// construct (e.g. LeftAnti join) is a valid plan even when no physical
// operator can execute it; the failure surfaces at emit time, not at
// construction time.
enum class NodeKind {
    Scan,
    Filter,
    Project,
    Aggregate,
    Sort,
    Limit,
    Join,
    Distinct,
    Concat,
};

// Join variants observed across TPC-H. Inner/LeftOuter are standard;
// LeftSemi and LeftAnti are how correlated EXISTS / NOT EXISTS is
// lowered (used in Q4, Q20, Q21, Q22); Cross appears in Q11 and Q15
// where an aggregated scalar is joined with every row for a threshold
// comparison. Right-outer / full-outer are not observed in TPC-H and
// are deferred until a real workload demands them.
enum class JoinKind {
    Inner,
    LeftOuter,
    LeftSemi,
    LeftAnti,
    Cross,
};

// Aggregate function kinds. COUNT(*) is distinct from COUNT(expr)
// because COUNT(*) doesn't need a body at all and ignores null inputs
// trivially; COUNT(DISTINCT expr) needs de-duplication semantics.
enum class AggKind {
    Sum,
    Avg,
    Min,
    Max,
    Count,
    CountAll,
    CountDistinct,
};

struct ProjectItem {
    std::string name;  // cosmetic; the Field id is what matters
    Expr expr;
};

struct AggregateItem {
    std::string name;
    AggKind agg;
    Expr body;  // ignored when agg == CountAll
};

struct SortKey {
    ColumnId id;
    gamax::sort_order order {gamax::sort_order::ASCENDING};
};

// Plan-tree node. Held as a shared_ptr so the IR can shape itself as a
// DAG: a single sub-plan may feed multiple parents (CTEs, self-joins,
// shared sub-expressions). Visitors must dedup by identity, and rule
// rewrites must clone-on-write if they'd mutate a multi-parent node.
//
// Per-kind payload lives inline as named fields. `kind` selects which
// payload is meaningful; the others are left default-constructed. The
// `output_schema` is computed and cached by the factory; it is the
// declared type of this node's output and is referenced by expressions
// in any parent node.
class PlanNode {
public:
    NodeKind kind;
    std::vector<std::shared_ptr<PlanNode>> inputs;
    Schema output_schema;

    // Scan payload
    std::string scan_table;
    std::vector<std::string> scan_columns;

    // Filter payload
    Expr filter_predicate;

    // Project payload
    std::vector<ProjectItem> project_items;

    // Aggregate payload
    std::vector<ColumnId> agg_keys;
    std::vector<AggregateItem> agg_items;

    // Sort payload
    std::vector<SortKey> sort_keys;

    // Limit payload
    gamax::size_type limit_n      {0};
    gamax::size_type limit_offset {0};

    // Join payload. `join_condition` is std::nullopt iff join_kind is
    // Cross (a Cartesian product has no condition); for all other kinds
    // it holds a BOOL8 predicate over the union of left + right
    // schemas.
    JoinKind join_kind {JoinKind::Inner};
    std::optional<Expr> join_condition;

    // Distinct payload. Empty `distinct_keys` means "distinct over all
    // output columns" (the SQL DISTINCT default).
    std::vector<ColumnId> distinct_keys;
};

using PlanPtr = std::shared_ptr<PlanNode>;

// ── Factories ─────────────────────────────────────────────────────
//
// Every factory:
//   1. validates the inputs (referenced ColumnIds live in the input
//      schema, expression types are well-formed, etc.),
//   2. derives the output_schema bottom-up,
//   3. returns a shared_ptr so plans can be shared as a DAG.
//
// Invalid constructions throw std::runtime_error at factory-call time;
// by design, a constructed PlanNode is always well-typed.

// Scan reads named columns from a catalog-registered table. Each
// requested column gets a fresh ColumnId from `gen`, so two scans of
// the same table are distinct column sources (no aliasing).
PlanPtr scan(
    const Catalog& catalog,
    std::string table_name,
    std::vector<std::string> columns,
    IdGenerator& gen = IdGenerator::default_instance());

// Filter keeps rows where `predicate` evaluates to true. The output
// schema is the input schema (no column identity changes — critical
// for pushdown algebra).
PlanPtr filter(PlanPtr input, Expr predicate);

// Project transforms each input row into a new row of named
// expressions. Pure ColRef projections preserve the input's ColumnId so
// that "project = rename / subset / reorder" does not renumber anything.
// Computed expressions get fresh ColumnIds from `gen`.
PlanPtr project(
    PlanPtr input,
    std::vector<ProjectItem> items,
    IdGenerator& gen = IdGenerator::default_instance());

// Aggregate groups by the given key ColumnIds and computes the given
// aggregates per group. Keys are carried forward into the output with
// their input ColumnIds (no rename); aggregate outputs get fresh ids
// from `gen`. An empty keys vector produces a scalar aggregate — the
// output is a single row. An empty aggs vector is not allowed here
// (use the dedicated `distinct` factory for key-only deduplication).
PlanPtr aggregate(
    PlanPtr input,
    std::vector<ColumnId> keys,
    std::vector<AggregateItem> aggs,
    IdGenerator& gen = IdGenerator::default_instance());

// Sort orders rows by the given keys. Preserves the input schema
// exactly (sort is a reorder, not a transform). Downstream nodes may
// rely on the induced collation for optimizations (merge-join, stream
// aggregation); that property is tracked on the `Collation` trait in
// a later phase, not here.
PlanPtr sort(PlanPtr input, std::vector<SortKey> keys);

// Limit truncates to the first `n` rows after an `offset`. Preserves
// the input schema. TPC-H uses LIMIT only with small n (top-K), which
// the backend's topk_table operator covers natively.
PlanPtr limit(PlanPtr input, gamax::size_type n, gamax::size_type offset = 0);

// Inner equi/theta join. `on` must be a BOOL8 predicate over the union
// of left and right output schemas. The output schema is left ++ right;
// no reindexing of ColumnIds occurs. Non-equi conditions are legal at
// the IR level; the emitter chooses an appropriate physical strategy
// (or errors if none is available).
PlanPtr inner_join(PlanPtr left, PlanPtr right, Expr on);

// Left outer join. Same as inner_join except all fields from `right`
// become nullable in the output schema (NULL-extended when no match).
PlanPtr left_outer_join(PlanPtr left, PlanPtr right, Expr on);

// Left semi-join: keeps each left row that has at least one matching
// right row, returning only left columns. This is how `EXISTS
// (correlated subquery)` lowers. Output schema == left schema.
PlanPtr left_semi_join(PlanPtr left, PlanPtr right, Expr on);

// Left anti-join: keeps left rows with NO matching right row. Lowers
// from `NOT EXISTS (correlated subquery)`. Output schema == left schema.
PlanPtr left_anti_join(PlanPtr left, PlanPtr right, Expr on);

// Cross join: Cartesian product. No condition. Output schema is
// left ++ right. Used in TPC-H Q11 / Q15 where a scalar aggregate is
// combined with every row of another stream for a threshold check.
PlanPtr cross_join(PlanPtr left, PlanPtr right);

// Distinct removes duplicate rows.
//   * Empty `keys` → dedup over every input column (SQL DISTINCT).
//     Output schema == input schema.
//   * `keys` covering every input column (by id) → identical result
//     to the empty case.
//   * `keys` is a non-empty proper subset → dedup by those keys; output
//     schema is just the key columns, in caller order. Matches the
//     `SELECT DISTINCT k1, k2 FROM t` shape — non-key columns are
//     dropped rather than getting an arbitrary pick per group. SQL
//     `DISTINCT ON (subset) keep all cols` is not expressible under
//     this factory; add a separate entry point when a caller needs it.
PlanPtr distinct(PlanPtr input, std::vector<ColumnId> keys = {});

// Vertical N-ary concatenation (SQL UNION ALL, once a frontend
// exposes it; today's only caller is build_q13's LEFT OUTER JOIN
// rewrite). Every input must have identical column counts and
// positionally matching types; nullability is OR'd across inputs.
// Output column names come from inputs[0]; output ColumnIds are
// freshly allocated so downstream references don't accidentally
// alias any specific input's ids. Throws on empty-inputs vector,
// arity mismatch, or type mismatch.
PlanPtr concat(
    std::vector<PlanPtr> inputs,
    IdGenerator& gen = IdGenerator::default_instance());

} // namespace gamax::ir
