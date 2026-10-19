// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "exec/executor.hpp"

#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

// Per-Value slot in the Executor. Three valid states:
//   * SourceTable:  view set (caller-supplied), owned null.
//   * Most steps:   owned set (produced by run_step), view default
//                   (empty); view_of returns owned->view().
//   * Reorder:      both set. owned holds an upstream table whose
//                   ownership we moved into this slot; view is a
//                   select_columns over those underlying columns.
//                   view_of returns view, which is the consumer-
//                   facing reordered shape. owned exists solely to
//                   pin the underlying memory so the column_views
//                   inside `view` stay valid until this slot is
//                   freed.
struct Slot {
    std::unique_ptr<gamax::table> owned;
    gamax::table_view view;
    std::shared_ptr<void> opaque;
};

// Prefer an explicit view (set by SourceTable or by an aliasing
// Reorder); fall back to owned->view() for the common owned-only
// case. An explicit view is identified by having ≥1 column — both
// SourceTable and Reorder always produce ≥1 column, and a
// default-constructed view has 0 columns.
gamax::table_view view_of(const Slot& s) {
    if (s.view.num_columns() > 0) return s.view;
    if (s.owned) return s.owned->view();
    return s.view;
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
                v.select_columns(pl.key_ordinals), ctx, pl.hint);
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
//
// `live` is mutable because Reorder is implemented as ownership
// transfer: the upstream's `owned` table is moved into the Reorder
// slot, leaving the upstream slot empty. Liveness analysis already
// guarantees the upstream has no other consumers after the Reorder
// step, so this is safe.
Slot step_dispatch(
    const ExecPlan& plan,
    gamax::size_type step_index,
    const Step& s,
    std::unordered_map<ValueId, Slot>& live,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx,
    const StepExtension* extension,
    const char* nvtx_prefix = "step")
{
    // Per-step NVTX range for the Nsight Systems timeline. Label
    // includes step index + kind name so sibling steps of the same
    // kind (two EqJoins, etc.) are distinguishable on the timeline.
    // Cheap enough to keep compiled in unconditionally.
    const std::string step_label = format_step_label(plan, step_index);
    std::string nvtx_label = std::string(nvtx_prefix) + "[" +
        std::to_string(step_index) + "]:" + step_label;
    nvtxRangePushA(nvtx_label.c_str());

    // Optional per-step GPU timing (opt-in via `ctx.profile_sink`).
    // SourceTable is included when the sink is wired; the bench
    // harness opts out by passing a sink-free ctx for phase 1.
    const bool profile = (ctx.profile_sink != nullptr);
    if (profile) {
        ctx.profile_sink->begin_step(
            static_cast<int>(step_index), s.kind, step_label, ctx.stream);
    }

    const bool handled_by_extension =
        extension && extension->handles && extension->handles(s.kind);
    std::unique_ptr<PhaseRange> local_phase;
    const bool phase_instrumentation =
        (ctx.profile_sink && ctx.profile_sink->collects_phases()) ||
        ctx.emit_phase_nvtx;
    if (phase_instrumentation && !handled_by_extension &&
        s.kind != StepKind::SourceTable) {
        local_phase = std::make_unique<PhaseRange>(
            ctx,
            PhaseMetadata{
                std::string("local.") + step_kind_name(s.kind),
                PhaseCategory::Compute});
    }

    Slot slot;
    if (handled_by_extension) {
        if (!extension->execute) {
            exec_error("step extension has no execute callback");
        }
        std::vector<ExtensionInput> inputs;
        inputs.reserve(s.inputs.size());
        for (const auto input_id : s.inputs) {
            auto it = live.find(input_id);
            if (it == live.end()) {
                exec_error(
                    "extension step missing input value " +
                    std::to_string(input_id));
            }
            inputs.push_back(ExtensionInput{view_of(it->second), it->second.opaque});
        }
        auto result = extension->execute(s, inputs, ctx);
        if (!result.table && !result.opaque) {
            exec_error("step extension returned neither table nor opaque value");
        }
        slot.owned = std::move(result.table);
        slot.opaque = std::move(result.opaque);
    } else if (s.kind == StepKind::SourceTable) {
        // SourceTable's "output" aliases caller-owned data; no
        // run_step dispatch — the resolver is the work.
        const auto& pl = std::get<SourceTablePayload>(s.payload);
        slot.view = source_resolver(pl.table_name, pl.columns);
    } else if (s.kind == StepKind::Reorder) {
        // Aliasing Reorder: skip the gather kernel that
        // `phys::reorder` would run and just take ownership of the
        // upstream's columns + present a `select_columns` view. At
        // SF=100 the original gather costs ~36 GiB of fresh
        // contiguous allocations (one full-cardinality column per
        // ordinal), and that single step is what OOMs Q1's slab.
        // Aliasing makes Reorder a zero-allocation pointer shuffle.
        //
        // Two correctness conditions:
        //
        // 1. select_columns must reference the upstream's *consumer-
        //    facing* shape. When the upstream is itself a Reorder
        //    slot, its `owned` holds the original un-reordered table
        //    moved up through the chain, while its `view` is the
        //    reordered shape downstream actually sees. Use view_of()
        //    so chained Reorders compose correctly.
        //
        // 2. We can only steal `owned` from the upstream if this
        //    Reorder is the upstream's *last* consumer. Plan DAGs
        //    that share a subexpression (e.g. a Project whose output
        //    feeds both a Reorder and another step) would crash with
        //    UAF if we moved out from under the other consumer.
        //    `last_used_step` aggregates over all consumers, so
        //    "Reorder is the last consumer" ⟺
        //    "upstream.last_used_step == this step's index". When
        //    that fails, fall back to the gather copy.
        const auto& pl = std::get<ReorderPayload>(s.payload);
        auto it = live.find(s.inputs.at(0));
        if (it == live.end()) exec_error(
            "Reorder: missing input value " +
            std::to_string(s.inputs.at(0)));
        const bool is_last_consumer =
            plan.values[s.inputs.at(0)].last_used_step == step_index;
        if (is_last_consumer) {
            gamax::table_view src_view = view_of(it->second);
            slot.view = src_view.select_columns(pl.ordinals);
            slot.owned = std::move(it->second.owned);
        } else {
            // Shared subexpression — keep the upstream alive for
            // its other consumers and pay the gather cost here.
            slot.owned = run_step(s, live, source_resolver, ctx);
        }
    } else if (s.kind == StepKind::Project) {
        // Mixed-alias Project: ColRef items pass through by *moving*
        // the matching column out of upstream's owned table; only
        // non-trivial items (Computed ExprTree, Predicate, Substring)
        // run through phys::project. Saves the redundant
        // copy_any_column on every passthrough column — at SF=100
        // this is multiple GiB per step on Q1-shaped wide projects.
        //
        // Conditions to take the alias path:
        //   1. Last consumer (DAG safety; mirrors Reorder).
        //   2. Upstream is pure-owned (`owned` set, `view` default-
        //      empty). SourceTable upstreams (no `owned`) and
        //      Reorder-aliased upstreams (explicit view) fall back.
        //      That naturally protects source-table data — Scan
        //      upstreams always copy because there's nothing to
        //      move from.
        //
        // Other subtleties:
        //   * If two ColRef items reference the same source ordinal,
        //     only the first can be moved; later refs go through
        //     phys::project's existing copy path. Tracked via a
        //     small seen-ordinals set.
        //   * phys::project must run on the *intact* upstream view —
        //     it reads source columns by ordinal, and a moved-out
        //     slot is nullptr. So we evaluate first, *then* move.
        const auto& pl = std::get<ProjectPayload>(s.payload);
        auto it = live.find(s.inputs.at(0));
        if (it == live.end()) exec_error(
            "Project: missing input value " +
            std::to_string(s.inputs.at(0)));
        const bool is_last_consumer =
            plan.values[s.inputs.at(0)].last_used_step == step_index;
        const bool upstream_pure_owned =
            it->second.owned && it->second.view.num_columns() == 0;

        if (is_last_consumer && upstream_pure_owned) {
            const auto n = pl.items.size();
            std::vector<gamax::column_ptr> output(n);

            // Pass 1: split items into "moved" (first-ref ColRef) vs
            // "remaining" (everything else, including duplicate
            // ColRefs and all non-trivial items).
            std::vector<gamax::phys::ProjectItem> remaining;
            std::vector<size_t> remaining_positions;
            std::vector<size_t> moved_positions;
            std::vector<gamax::size_type> moved_ordinals;
            std::unordered_set<gamax::size_type> seen_ordinals;
            remaining.reserve(n);
            remaining_positions.reserve(n);
            moved_positions.reserve(n);
            moved_ordinals.reserve(n);

            for (size_t i = 0; i < n; ++i) {
                const auto& item = pl.items[i];
                if (std::holds_alternative<gamax::expression::ExprTree>(item)) {
                    const auto& e =
                        std::get<gamax::expression::ExprTree>(item);
                    if (e.is_column_ref()) {
                        const auto src = e.get_column_index();
                        if (seen_ordinals.insert(src).second) {
                            moved_positions.push_back(i);
                            moved_ordinals.push_back(src);
                            continue;
                        }
                    }
                }
                remaining.push_back(item);
                remaining_positions.push_back(i);
            }

            // Pass 2: evaluate the non-trivial items via phys::project
            // on the *intact* upstream view. Run before any moves so
            // the view sees all source columns.
            if (!remaining.empty()) {
                gamax::table_view in_view = it->second.owned->view();
                auto sub = gamax::phys::project(
                    in_view, remaining, ctx, pl.hint);
                for (size_t k = 0; k < remaining_positions.size(); ++k) {
                    output[remaining_positions[k]] = sub->remove_column(0);
                }
            }

            // Pass 3: move ColRef columns out of upstream's owned
            // table into their target output positions. release_column
            // leaves a nullptr behind, preserving the column indices
            // that earlier-evaluated kernels (in pass 2) referenced.
            for (size_t k = 0; k < moved_positions.size(); ++k) {
                output[moved_positions[k]] =
                    it->second.owned->release_column(moved_ordinals[k]);
            }

            slot.owned = std::make_unique<gamax::table>(std::move(output));
        } else {
            // Shared subexpression or non-pure-owned upstream — fall
            // back to the original full-copy phys::project behaviour.
            slot.owned = run_step(s, live, source_resolver, ctx);
        }
    } else {
        slot.owned = run_step(s, live, source_resolver, ctx);
    }

    local_phase.reset();
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
            if (it != live.end() && (it->second.owned || it->second.opaque)) {
                live.erase(it);
            }
        }
    }
}

// Pull the final result out of the live map. Used by execute() and
// bench_run_steps(). When the final slot is a non-owning view (pure-
// Scan plan ending at a SourceTable, or a plan whose final step is
// an aliasing Reorder), materialize the view's column selection into
// a fresh owned table so the caller gets a standalone result.
std::unique_ptr<gamax::table> finalize_result(
    const ExecPlan& plan,
    std::unordered_map<ValueId, Slot>& live,
    const gamax::phys::Context& ctx)
{
    auto final_id = plan.steps.back().output;
    auto it = live.find(final_id);
    if (it == live.end()) exec_error("final value missing");
    auto& slot = it->second;
    // Aliasing Reorder slot: explicit view + (possibly) owned. The
    // view is the user-facing shape; owned only pins memory. We
    // can't return owned directly because its column layout is the
    // upstream's, not the reorder selection. Materialize via gather
    // so the caller's table has the expected shape.
    if (slot.view.num_columns() > 0) {
        auto v = slot.view;
        std::vector<gamax::size_type> identity(v.num_columns());
        std::iota(identity.begin(), identity.end(), gamax::size_type{0});
        return gamax::phys::reorder(v, std::move(identity), ctx);
    }
    if (slot.owned) return std::move(slot.owned);
    exec_error("final value has neither owned nor view");
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// Production entry point: walk the plan, dispatch every step,
// aggressively free dead inputs.
// ─────────────────────────────────────────────────────────────────────
std::unique_ptr<gamax::table> execute_impl(
    const ExecPlan& plan,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx,
    const StepExtension* extension)
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
        live[s.output] = step_dispatch(
            plan, i, s, live, source_resolver, ctx, extension, "step");

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

std::unique_ptr<gamax::table> execute(
    const ExecPlan& plan,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx)
{
    return execute_impl(plan, source_resolver, ctx, nullptr);
}

std::unique_ptr<gamax::table> execute(
    const ExecPlan& plan,
    const SourceResolver& source_resolver,
    const gamax::phys::Context& ctx,
    const StepExtension& extension)
{
    return execute_impl(plan, source_resolver, ctx, &extension);
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
        live[s.output] = step_dispatch(
            plan, i, s, live, source_resolver, ctx, nullptr, "load.step");
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
std::unique_ptr<gamax::table> bench_run_steps_impl(
    const ExecPlan& plan,
    const BenchExecState& state,
    const gamax::phys::Context& ctx,
    const StepExtension* extension)
{
    if (plan.steps.empty()) exec_error("empty plan");

    std::unordered_map<ValueId, Slot> live;
    live.reserve(state.impl()->live.size() + plan.values.size());
    for (const auto& [value_id, source] : state.impl()->live) {
        if (source.owned || source.opaque) {
            exec_error("benchmark preload state contains a non-source value");
        }
        Slot copy;
        copy.view = source.view;
        live.emplace(value_id, std::move(copy));
    }

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
        live[s.output] = step_dispatch(
            plan, i, s, live, null_resolver, ctx, extension, "step");
        free_dead_inputs(plan, s, i, live);
    }

    return finalize_result(plan, live, ctx);
}

std::unique_ptr<gamax::table> bench_run_steps(
    const ExecPlan& plan,
    const BenchExecState& state,
    const gamax::phys::Context& ctx)
{
    return bench_run_steps_impl(plan, state, ctx, nullptr);
}

std::unique_ptr<gamax::table> bench_run_steps(
    const ExecPlan& plan,
    const BenchExecState& state,
    const gamax::phys::Context& ctx,
    const StepExtension& extension)
{
    return bench_run_steps_impl(plan, state, ctx, &extension);
}

} // namespace gamax::exec
