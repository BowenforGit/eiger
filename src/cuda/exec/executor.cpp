#include "exec/executor.hpp"

#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include "exec/print.hpp"
#include "exec/step_profile.hpp"

#ifdef GAMAX_PHYS_DEBUG
#include <cctype>
#include <iostream>
#include "utils/debug_print.hpp"
#endif

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
            auto items = pl.items;
            return gamax::phys::project(input_view(0), items, ctx, pl.hint);
        } else if constexpr (std::is_same_v<T, FilterPayload>) {
            auto pred = pl.predicate;  // non-const needed
            if (pl.output_ordinals.empty()) {
                return gamax::phys::filter(input_view(0), pred, ctx, pl.hint);
            }
            return gamax::phys::filter(
                input_view(0), pred, pl.output_ordinals, ctx, pl.hint);
        } else if constexpr (std::is_same_v<T, EqJoinPayload>) {
            return gamax::phys::eq_join(
                input_view(0), pl.lkeys,
                input_view(1), pl.rkeys, ctx, pl.hint);
        } else if constexpr (std::is_same_v<T, SemiJoinPayload>) {
            return gamax::phys::semi_join(
                input_view(0), pl.lkeys,
                input_view(1), pl.rkeys, ctx, pl.hint);
        } else if constexpr (std::is_same_v<T, AntiJoinPayload>) {
            return gamax::phys::anti_join(
                input_view(0), pl.lkeys,
                input_view(1), pl.rkeys, ctx, pl.hint);
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
                pl.specs, ctx, pl.hint);
        } else if constexpr (std::is_same_v<T, DistinctPayload>) {
            // Sparse column selection: the input may carry more than
            // just the distinct keys, but only `pl.key_ordinals` are
            // handed to `phys::distinct` as keys.
            auto v = input_view(0);
            return gamax::phys::distinct(
                v.select_columns(pl.key_ordinals), ctx);
        } else if constexpr (std::is_same_v<T, ConcatPayload>) {
            // N-ary: one view per input ValueId on Step::inputs.
            std::vector<gamax::table_view> views;
            views.reserve(s.inputs.size());
            for (size_t i = 0; i < s.inputs.size(); ++i) {
                views.push_back(input_view(i));
            }
            return gamax::phys::concat(std::move(views), ctx);
        } else if constexpr (std::is_same_v<T, SortPayload>) {
            return gamax::phys::sort(input_view(0), pl.keys, ctx, pl.hint);
        } else if constexpr (std::is_same_v<T, TopKPayload>) {
            return gamax::phys::top_k(
                input_view(0), pl.keys, pl.k, ctx, pl.hint);
        }
        exec_error("unknown StepPayload alternative");
    }, s.payload);
}

// Run one step into a Slot, with NVTX + StepProfileSink wrapping. The
// caller is responsible for storing the returned Slot into `live` and
// for any post-step bookkeeping (debug hook, freeing dead inputs).
//
// Both the production `execute()` path and the two-phase bench path
// dispatch through this helper, so the per-step instrumentation lives
// in exactly one place.
Slot step_dispatch(
    gamax::size_type step_index,
    const Step& s,
    const std::unordered_map<ValueId, Slot>& live,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx)
{
    // Per-step NVTX range for the Nsight Systems timeline. Label
    // includes step index + kind name so sibling steps of the same
    // kind (two EqJoins, etc.) are distinguishable on the timeline.
    // Cheap enough to keep compiled in unconditionally.
    std::string nvtx_label =
        "step[" + std::to_string(step_index) + "]:" +
        step_kind_name(s.kind);
    nvtxRangePushA(nvtx_label.c_str());

    // Optional per-step GPU timing (opt-in via `ctx.profile_sink`).
    // SourceTable is included when the sink is wired; the bench
    // harness opts out by passing a sink-free ctx for phase 1.
    const bool profile = (ctx.profile_sink != nullptr);
    if (profile) {
        ctx.profile_sink->begin_step(
            static_cast<int>(step_index), s.kind, ctx.stream);
    }

    Slot slot;
    if (s.kind == StepKind::SourceTable) {
        // SourceTable's "output" aliases caller-owned data; no
        // run_step dispatch — the resolver is the work.
        const auto& pl = std::get<SourceTablePayload>(s.payload);
        slot.view = source_resolver(pl.table_name, pl.columns);
    } else {
        slot.owned = run_step(s, live, source_resolver, ctx);
    }

    if (profile) ctx.profile_sink->end_step(ctx.stream);
    nvtxRangePop();
    return slot;
}

// Drop owned slots whose `last_used_step` has passed. Source-table
// slots (non-owning) are kept regardless — they alias caller data and
// freeing them would leave dangling views in any later step that still
// reads from the same source.
void free_dead_inputs(
    const ExecPlan& plan,
    const Step& s,
    gamax::size_type step_index,
    std::unordered_map<ValueId, Slot>& live)
{
    for (auto in : s.inputs) {
        if (plan.values[in].last_used_step <= step_index) {
            auto it = live.find(in);
            if (it != live.end() && it->second.owned) {
                live.erase(it);
            }
        }
    }
}

// Pull the final result out of the live map. Used by execute() and
// bench_run_steps(). For pure-Scan plans (final value is a non-owning
// SourceTable view), materialize via identity reorder so the caller
// receives a standalone owned table.
std::unique_ptr<gamax::table> finalize_result(
    const ExecPlan& plan,
    std::unordered_map<ValueId, Slot>& live,
    const gamax::phys::Context& ctx)
{
    auto final_id = plan.steps.back().output;
    auto it = live.find(final_id);
    if (it == live.end()) exec_error("final value missing");
    if (it->second.owned) return std::move(it->second.owned);
    auto v = it->second.view;
    std::vector<gamax::size_type> identity(v.num_columns());
    std::iota(identity.begin(), identity.end(), gamax::size_type{0});
    return gamax::phys::reorder(v, std::move(identity), ctx);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// Production entry point: walk the plan, dispatch every step,
// aggressively free dead inputs.
// ─────────────────────────────────────────────────────────────────────
std::unique_ptr<gamax::table> execute(
    const ExecPlan& plan,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx)
{
    if (plan.steps.empty()) exec_error("empty plan");

    std::unordered_map<ValueId, Slot> live;
    live.reserve(plan.values.size());

#ifdef GAMAX_PHYS_DEBUG
    // Per-execute() env-var snapshot. Parses GAMAX_PHYS_DEBUG (a comma
    // list of StepKind names or `all`) and GAMAX_PHYS_DEBUG_ROWS. When
    // neither names nor `all` are set, the filter admits nothing and
    // the loop below skips the per-step print work entirely.
    gamax::util::debug::PhysDebugFilter phys_debug;
#endif

    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(plan.steps.size()); ++i)
    {
        const auto& s = plan.steps[i];
        live[s.output] = step_dispatch(i, s, live, source_resolver, ctx);

#ifdef GAMAX_PHYS_DEBUG
        // Debug hook — fires for *every* step (including SourceTable)
        // when the user has enabled a matching kind via the
        // GAMAX_PHYS_DEBUG env var. We normalise the kind name to
        // lower-case so the matching is case-insensitive against the
        // camel-case names returned by `step_kind_name`.
        if (phys_debug.any_enabled()) {
            std::string kind_lower = step_kind_name(s.kind);
            for (auto& c : kind_lower) {
                c = static_cast<char>(std::tolower(
                    static_cast<unsigned char>(c)));
            }
            if (phys_debug.should_print(kind_lower)) {
                // Sync the ctx stream before the device→host reads in
                // print_table; otherwise in-flight kernels could race
                // with the cudaMemcpys.
                cudaStreamSynchronize(ctx.stream);
                auto v = view_of(live[s.output]);
                std::cerr << "[step " << i << ": "
                          << step_kind_name(s.kind) << "] -> "
                          << v.num_rows() << " rows x "
                          << v.num_columns() << " cols\n";
                gamax::util::debug::print_table(
                    v, phys_debug.max_rows(), /*offset=*/0, std::cerr);
                // Blank line between step blocks for readability.
                std::cerr << "\n";
                std::cerr.flush();
            }
        }
#endif

        free_dead_inputs(plan, s, i, live);
    }

    return finalize_result(plan, live, ctx);
}

// ─────────────────────────────────────────────────────────────────────
// BenchExecState — opaque carrier for the populated `live` map between
// bench_resolve_sources() and bench_run_steps(). Implemented as PIMPL
// because `Slot` is private to this translation unit.
// ─────────────────────────────────────────────────────────────────────
struct BenchExecState::Impl {
    std::unordered_map<ValueId, Slot> live;
};

BenchExecState::BenchExecState() : impl_(std::make_unique<Impl>()) {}
BenchExecState::~BenchExecState() = default;
BenchExecState::BenchExecState(BenchExecState&&) noexcept = default;
BenchExecState& BenchExecState::operator=(BenchExecState&&) noexcept = default;

// ─────────────────────────────────────────────────────────────────────
// Phase 1 of the bench path: resolve every SourceTable step into the
// live map. Untimed in the bench fixture; the resolver's per-cell
// cache means rep-0 pays the H2D and subsequent reps hit the cache.
// We sync the stream on return so any in-flight async copies (e.g.
// pinned→device done by the resolver) are complete before phase 2.
// ─────────────────────────────────────────────────────────────────────
BenchExecState bench_resolve_sources(
    const ExecPlan& plan,
    const SourceResolver& source_resolver,
    cudaStream_t stream)
{
    if (plan.steps.empty()) exec_error("empty plan");

    BenchExecState state;
    auto& live = state.impl()->live;
    live.reserve(plan.values.size());

    // Phase 1's ctx carries no profile_sink — SourceTable timings
    // belong outside the breakdown by construction here. The ctx is
    // passed only so step_dispatch's signature stays uniform; the
    // SourceTable branch does not actually invoke any phys ops.
    gamax::phys::Context ctx{stream, /*profile_sink=*/nullptr};

    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(plan.steps.size()); ++i)
    {
        const auto& s = plan.steps[i];
        if (s.kind != StepKind::SourceTable) continue;
        live[s.output] = step_dispatch(i, s, live, source_resolver, ctx);
    }

    cudaStreamSynchronize(stream);
    return state;
}

// ─────────────────────────────────────────────────────────────────────
// Phase 2 of the bench path: walk the plan, skip SourceTable steps,
// dispatch the rest into the pre-built live map. Honors the same
// last_used_step freeing as execute(). NVBench should wrap this call
// in `timer.start()`/`timer.stop()`.
//
// The SourceResolver passed to step_dispatch is a sentinel that throws
// if invoked — phase 2 must never see a SourceTable step.
// ─────────────────────────────────────────────────────────────────────
std::unique_ptr<gamax::table> bench_run_steps(
    const ExecPlan& plan,
    BenchExecState state,
    const gamax::phys::Context& ctx)
{
    if (plan.steps.empty()) exec_error("empty plan");

    auto& live = state.impl()->live;

    SourceResolver null_resolver =
        [](const std::string&,
           const std::vector<std::string>&) -> gamax::table_view {
            exec_error("bench_run_steps: SourceTable seen in phase 2");
        };

    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(plan.steps.size()); ++i)
    {
        const auto& s = plan.steps[i];
        if (s.kind == StepKind::SourceTable) continue;
        live[s.output] = step_dispatch(i, s, live, null_resolver, ctx);
        free_dead_inputs(plan, s, i, live);
    }

    return finalize_result(plan, live, ctx);
}

} // namespace gamax::exec
