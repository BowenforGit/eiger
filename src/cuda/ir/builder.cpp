#include "ir/builder.hpp"

#include <stdexcept>

namespace gamax::ir {

namespace {
[[noreturn]] void builder_error(const std::string& what) {
    throw std::runtime_error("ir::Builder: " + what);
}
} // namespace

ColumnId PlanBuilder::resolve(const std::string& name) const {
    auto idx = _plan->output_schema.index_of(name);
    if (!idx) builder_error("column '" + name + "' not in current schema");
    return _plan->output_schema[*idx].id;
}

Expr PlanBuilder::c(const std::string& name) const {
    auto idx = _plan->output_schema.index_of(name);
    if (!idx) builder_error("column '" + name + "' not in current schema");
    const auto& f = _plan->output_schema[*idx];
    return col_ref(f.id, f.type, f.nullable);
}

PlanBuilder PlanBuilder::filter(Expr predicate) const {
    return PlanBuilder(gamax::ir::filter(_plan, std::move(predicate)), _gen);
}

PlanBuilder PlanBuilder::project(std::vector<ProjectItem> items) const {
    return PlanBuilder(
        gamax::ir::project(_plan, std::move(items), *_gen), _gen);
}

PlanBuilder PlanBuilder::aggregate(
    std::vector<std::string> group_by_names,
    std::vector<AggregateItem> aggs) const
{
    std::vector<ColumnId> keys;
    keys.reserve(group_by_names.size());
    for (const auto& n : group_by_names) keys.push_back(resolve(n));
    return PlanBuilder(
        gamax::ir::aggregate(_plan, std::move(keys), std::move(aggs), *_gen),
        _gen);
}

PlanBuilder PlanBuilder::sort(
    std::vector<std::pair<std::string, gamax::sort_order>> keys) const
{
    std::vector<SortKey> resolved;
    resolved.reserve(keys.size());
    for (auto& k : keys) resolved.push_back({resolve(k.first), k.second});
    return PlanBuilder(
        gamax::ir::sort(_plan, std::move(resolved)), _gen);
}

PlanBuilder PlanBuilder::limit(
    gamax::size_type n, gamax::size_type offset) const
{
    return PlanBuilder(gamax::ir::limit(_plan, n, offset), _gen);
}

PlanBuilder PlanBuilder::distinct(std::vector<std::string> keys) const {
    std::vector<ColumnId> resolved;
    resolved.reserve(keys.size());
    for (const auto& n : keys) resolved.push_back(resolve(n));
    return PlanBuilder(
        gamax::ir::distinct(_plan, std::move(resolved)), _gen);
}

PlanBuilder PlanBuilder::inner_join(const PlanBuilder& right, Expr on) const {
    return PlanBuilder(
        gamax::ir::inner_join(_plan, right._plan, std::move(on)), _gen);
}
PlanBuilder PlanBuilder::left_outer_join(
    const PlanBuilder& right, Expr on) const
{
    return PlanBuilder(
        gamax::ir::left_outer_join(_plan, right._plan, std::move(on)), _gen);
}
PlanBuilder PlanBuilder::left_semi_join(
    const PlanBuilder& right, Expr on) const
{
    return PlanBuilder(
        gamax::ir::left_semi_join(_plan, right._plan, std::move(on)), _gen);
}
PlanBuilder PlanBuilder::left_anti_join(
    const PlanBuilder& right, Expr on) const
{
    return PlanBuilder(
        gamax::ir::left_anti_join(_plan, right._plan, std::move(on)), _gen);
}
PlanBuilder PlanBuilder::cross_join(const PlanBuilder& right) const {
    return PlanBuilder(gamax::ir::cross_join(_plan, right._plan), _gen);
}
PlanBuilder PlanBuilder::concat(const PlanBuilder& other) const {
    return PlanBuilder(
        gamax::ir::concat({_plan, other._plan}, *_gen), _gen);
}

PlanBuilder QueryBuilder::scan(
    std::string table, std::vector<std::string> columns) const
{
    auto p = gamax::ir::scan(*_catalog, std::move(table), std::move(columns),
                             *_gen);
    return PlanBuilder(std::move(p), _gen);
}

} // namespace gamax::ir
