#include "exec/executor.hpp"

#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace gamax::exec {

namespace {

[[noreturn]] void exec_error(const std::string& what) {
    throw std::runtime_error("exec::execute: " + what);
}

// Per-Value slot in the Executor. Holds either a caller-supplied
// `table_view` (for SourceTable steps, whose `owned` is null) or an
// owned `table_ptr` (for every other step).
struct Slot {
    std::unique_ptr<gamax::table> owned;
    gamax::table_view view;
};

gamax::table_view view_of(const Slot& s) {
    return s.owned ? s.owned->view() : s.view;
}

// Dispatch one step to the right `phys::*` call.
std::unique_ptr<gamax::table> run_step(
    const Step& s,
    const std::unordered_map<ValueId, Slot>& live,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx)
{
    auto input_view = [&](size_t i) -> gamax::table_view {
        auto it = live.find(s.inputs.at(i));
        if (it == live.end()) exec_error(
            "missing input value " + std::to_string(s.inputs.at(i)));
        return view_of(it->second);
    };

    // Note: `std::visit` on the variant payload dispatches per-kind.
    // We capture `s.inputs` / `input_view` by reference for the join
    // cases that take two inputs.
    return std::visit([&](auto&& pl) -> std::unique_ptr<gamax::table> {
        using T = std::decay_t<decltype(pl)>;
        if constexpr (std::is_same_v<T, SourceTablePayload>) {
            // SourceTable is handled specially by the caller (see
            // execute() — we short-circuit before reaching here).
            exec_error("SourceTable step should be handled outside run_step");
        } else if constexpr (std::is_same_v<T, ReorderPayload>) {
            return gamax::phys::reorder(input_view(0), pl.ordinals, ctx);
        } else if constexpr (std::is_same_v<T, ProjectPayload>) {
            // phys::project takes non-const ref; give it a local copy.
            auto exprs = pl.exprs;
            return gamax::phys::project(input_view(0), exprs, ctx);
        } else if constexpr (std::is_same_v<T, FilterPayload>) {
            auto pred = pl.predicate;  // non-const needed
            return gamax::phys::filter(input_view(0), pred, ctx);
        } else if constexpr (std::is_same_v<T, EqJoinPayload>) {
            return gamax::phys::eq_join(
                input_view(0), pl.lkeys,
                input_view(1), pl.rkeys, ctx);
        } else if constexpr (std::is_same_v<T, SemiJoinPayload>) {
            return gamax::phys::semi_join(
                input_view(0), pl.lkeys,
                input_view(1), pl.rkeys, ctx);
        } else if constexpr (std::is_same_v<T, CrossJoinPayload>) {
            return gamax::phys::cross_join(
                input_view(0), input_view(1), ctx);
        } else if constexpr (std::is_same_v<T, ScalarAggPayload>) {
            return gamax::phys::scalar_aggregate(
                input_view(0), pl.specs, ctx);
        } else if constexpr (std::is_same_v<T, GroupByPayload>) {
            // Split the materialized table into key and value views
            // at the boundary captured at lowering time.
            auto v = input_view(0);
            std::vector<gamax::size_type> k_indices(pl.n_keys);
            std::iota(k_indices.begin(), k_indices.end(), gamax::size_type{0});
            std::vector<gamax::size_type> v_indices(
                v.num_columns() - pl.n_keys);
            std::iota(v_indices.begin(), v_indices.end(), pl.n_keys);
            return gamax::phys::group_by(
                v.select_columns(k_indices),
                v.select_columns(v_indices),
                pl.specs, ctx);
        } else if constexpr (std::is_same_v<T, SortPayload>) {
            return gamax::phys::sort(input_view(0), pl.keys, ctx);
        } else if constexpr (std::is_same_v<T, TopKPayload>) {
            return gamax::phys::top_k(
                input_view(0), pl.keys, pl.k, ctx);
        }
        exec_error("unknown StepPayload alternative");
    }, s.payload);
}

} // namespace

std::unique_ptr<gamax::table> execute(
    const ExecPlan& plan,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx)
{
    if (plan.steps.empty()) exec_error("empty plan");

    std::unordered_map<ValueId, Slot> live;
    live.reserve(plan.values.size());

    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(plan.steps.size()); ++i)
    {
        const auto& s = plan.steps[i];

        // SourceTable is special: the "output" aliases a caller-owned
        // view, not a freshly-produced table. The Executor keeps the
        // view alive in its Slot without taking ownership.
        if (s.kind == StepKind::SourceTable) {
            const auto& pl = std::get<SourceTablePayload>(s.payload);
            Slot slot;
            slot.view = source_resolver(pl.table_name);
            live[s.output] = std::move(slot);
        } else {
            Slot slot;
            slot.owned = run_step(s, live, source_resolver, ctx);
            live[s.output] = std::move(slot);
        }

        // Drop any input whose last_used_step has passed. Don't drop
        // Source views — they alias caller-owned data.
        for (auto in : s.inputs) {
            if (plan.values[in].last_used_step <= i) {
                auto it = live.find(in);
                if (it != live.end() && it->second.owned) {
                    live.erase(it);
                }
            }
        }
    }

    auto final_id = plan.steps.back().output;
    auto it = live.find(final_id);
    if (it == live.end() || !it->second.owned) {
        exec_error("final value missing or not owned (SourceTable as root?)");
    }
    return std::move(it->second.owned);
}

} // namespace gamax::exec
