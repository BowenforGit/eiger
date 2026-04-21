#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ir/catalog.hpp"
#include "ir/column_id.hpp"
#include "ir/expr.hpp"
#include "ir/plan_node.hpp"
#include "ir/schema.hpp"
#include "types/data_type.hpp"

namespace gamax::ir {

// Fluent builder around a PlanPtr. Wraps the underlying PlanNode
// factories so that:
//   * the IdGenerator is threaded automatically,
//   * expressions reference columns by **name** (resolved against the
//     current output schema to a ColumnId), eliminating the `col0 /
//     col1` ordinal conventions used by the imperative TPC-H runners,
//   * schema and ColumnId-identity propagation stays invisible to the
//     caller unless they want to drop into the lower level.
//
// Builder methods are const and return a new PlanBuilder. The
// underlying plan is held by shared_ptr so chaining is cheap; sharing
// a PlanBuilder across branches yields a DAG naturally (e.g. self-join:
// `li.inner_join(li, ...)`).
class PlanBuilder {
public:
    PlanBuilder() = default;
    PlanBuilder(PlanPtr plan, IdGenerator* gen)
        : _plan(std::move(plan)), _gen(gen) {}

    // ── Accessors ──

    const PlanPtr& plan() const { return _plan; }
    const Schema& schema() const { return _plan->output_schema; }
    IdGenerator& id_generator() const { return *_gen; }

    // Column reference by name. Resolves against the current output
    // schema (which changes with each operator) and returns a typed
    // Expr leaf. Throws if the name isn't present.
    //
    // When two columns share a name (legal after a join), the *first*
    // match is returned. Callers that need a specific side should
    // capture the builder pre-join and use that.
    Expr c(const std::string& name) const;

    // ── Chaining ──

    PlanBuilder filter(Expr predicate) const;
    PlanBuilder project(std::vector<ProjectItem> items) const;
    PlanBuilder aggregate(std::vector<std::string> group_by_names,
                          std::vector<AggregateItem> aggs) const;
    PlanBuilder sort(std::vector<std::pair<std::string, gamax::sort_order>>
                         keys) const;
    PlanBuilder limit(gamax::size_type n,
                      gamax::size_type offset = 0) const;
    PlanBuilder distinct(std::vector<std::string> keys = {}) const;

    PlanBuilder inner_join(const PlanBuilder& right, Expr on) const;
    PlanBuilder left_outer_join(const PlanBuilder& right, Expr on) const;
    PlanBuilder left_semi_join(const PlanBuilder& right, Expr on) const;
    PlanBuilder left_anti_join(const PlanBuilder& right, Expr on) const;
    PlanBuilder cross_join(const PlanBuilder& right) const;

private:
    // Resolves `name` within `_plan->output_schema` to its ColumnId or
    // throws. Factored out because multiple methods need it.
    ColumnId resolve(const std::string& name) const;

    PlanPtr _plan;
    IdGenerator* _gen {nullptr};
};

// Top-level entry point. Owns the catalog reference and the id
// generator pointer used by every builder chain it produces.
class QueryBuilder {
public:
    QueryBuilder(const Catalog& catalog,
                 IdGenerator& gen = IdGenerator::default_instance())
        : _catalog(&catalog), _gen(&gen) {}

    // Begin a plan by scanning a catalog table. `columns` is an ordered
    // subset of that table's columns; each gets a fresh ColumnId.
    PlanBuilder scan(std::string table,
                     std::vector<std::string> columns) const;

    IdGenerator& id_generator() const { return *_gen; }
    const Catalog& catalog() const { return *_catalog; }

private:
    const Catalog* _catalog;
    IdGenerator* _gen;
};

// Convenience constructors for SortSpec-style pairs.
inline std::pair<std::string, gamax::sort_order> asc(std::string name) {
    return {std::move(name), gamax::sort_order::ASCENDING};
}
inline std::pair<std::string, gamax::sort_order> desc(std::string name) {
    return {std::move(name), gamax::sort_order::DESCENDING};
}

} // namespace gamax::ir
