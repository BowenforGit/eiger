// IR-driven TPC-H benchmark using NVBench's manual-timer mode.
//
// Methodology:
//   * SourceTable resolution (Parquet decode + pinned→device H2D) is
//     pulled out of the timed region by splitting `execute()` into
//     `bench_resolve_sources` (phase 1, untimed) and `bench_run_steps`
//     (phase 2, timed). The doc's must-have #2 ("Table scan shouldn't
//     be included in reporting total query execution time") is honored
//     by construction.
//   * NVBench owns the sweep / convergence / repetition / output. We
//     register one benchmark with `(query, scale)` axes; per-cell, the
//     fixture builds the IR plan once, then NVBench drives the manual
//     timer across multiple samples for stable timing.
//   * Reader sharing (must-have #6): a process-wide static map of
//     `TpchTableReader` keyed by scale factor preserves the per-column
//     pinned-host cache and the parquet::FileReader handles across all
//     cells. The IrSourceResolver itself is per-cell; its device-table
//     cache lives only across reps within one cell.
//
// CLI examples:
//   ./benchmark_tpch_ir                            # full sweep
//   ./benchmark_tpch_ir -a query=6 -a scale=10     # single cell
//   ./benchmark_tpch_ir --profile -a query=6       # one-shot, no timing
//                                                  # (attach `nsys profile`)
//   ./benchmark_tpch_ir -a "filter=[fuse]" -a "group_by=[hash,sort_gftr]"
//                                                  # operator-hint sweep
//
// Operator axes:
//   filter, project   "fuse"   "nofuse"   "n/a"
//   sort              "skf"    "no_skf"   "n/a"
//   group_by          "hash"   "sort_gftr"   "sort_gftr_opt"   "sort_gfur"
//                     "partition_gftr"   "partition_gfur"   "n/a"
//   join              "hash_gfur"   "hash_gftr"   "sort_merge_gfur"
//                     "sort_merge_gftr"   "partition_hash_gfur"
//                     "partition_hash_gftr"   "n/a"
//
// The "join" axis is unified across EqJoin / SemiJoin / AntiJoin —
// all join steps in a query share one hint per measurement, matching
// the per-kind-shares-one-config simplification.
//
// Each axis includes "n/a" as a sentinel. The fixture skips a cell if
// (axis="n/a" AND the corresponding StepKind appears in the lowered
// plan) OR (axis≠"n/a" AND the StepKind doesn't appear). That collapses
// the Cartesian to per-query valid combinations without the user
// writing per-query axis manifests. The "all instances of one
// operator kind share a single hint" simplification drops the
// per-instance axis explosion.
//
// Output: NVBench's standard CSV/JSON/Markdown via `--csv`, `--json`,
// `--md` flags. Per-step breakdown is intentionally not emitted by
// this binary — that's a separate side-channel via run_tpch_ir's
// `--profile-csv`.

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "exec/executor.hpp"
#include "exec/step_profile.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "ir/rules.hpp"
#include "memory/mem_manager.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/sort/sort.hpp"
#include "operators/select/select.hpp"
#include "operators/project/project.hpp"
#include "phys/phys.hpp"
#include "tpch/ir_source_resolver.hpp"
#include "tpch/tpch.hpp"
#include "tpch/tpch_ir.hpp"

namespace {

// Process-wide reader cache, keyed by scale factor. The reader holds
// the per-column pinned-host cache and the parquet::FileReader handle;
// neither is GPU memory, so keeping it alive for the full sweep costs
// only host RAM. Lazily initialized — a sweep that filters to one
// scale only opens that scale's readers.
gamax::tpch::TpchTableReader& shared_reader(int scale_factor) {
    static std::unordered_map<int,
        std::unique_ptr<gamax::tpch::TpchTableReader>> readers;
    auto it = readers.find(scale_factor);
    if (it == readers.end()) {
        gamax::tpch::tpch_params params(scale_factor);
        auto reader =
            std::make_unique<gamax::tpch::TpchTableReader>(params);
        it = readers.emplace(scale_factor, std::move(reader)).first;
    }
    return *it->second;
}

// Process-wide catalog. Schema is stable across queries and scales.
const gamax::ir::Catalog& shared_catalog() {
    static auto cat = gamax::ir::tpch::make_tpch_catalog();
    return cat;
}

// ── Axis parsing ──────────────────────────────────────────────────
//
// Each operator-kind axis is a string axis whose values include
// "n/a" plus the algorithm names recognized by
// `gamax::tpch::tpch_params::parse_algo_value`. Returning
// `std::nullopt` from these helpers signals the "n/a" sentinel — the
// fixture uses that to drive its skip-by-n/a logic.

constexpr const char* kNA = "n/a";

std::optional<gamax::select::select_options>
parse_filter_axis(const std::string& v) {
    if (v == kNA) return std::nullopt;
    gamax::select::select_options o;
    o.fuse_operations = (v == "fuse");
    return o;
}

std::optional<gamax::project::project_options>
parse_project_axis(const std::string& v) {
    if (v == kNA) return std::nullopt;
    gamax::project::project_options o;
    o.fuse_operations = (v == "fuse");
    return o;
}

std::optional<gamax::sort::sort_table_options>
parse_sort_axis(const std::string& v) {
    if (v == kNA) return std::nullopt;
    gamax::sort::sort_table_options o;
    o.smart_keys_fusion = (v == "skf");
    return o;
}

std::optional<gamax::groupby::groupby_hint>
parse_group_by_axis(const std::string& v) {
    if (v == kNA) return std::nullopt;
    int algo_int = gamax::tpch::tpch_params::parse_algo_value(v);
    gamax::groupby::groupby_hint h;
    h.groupby_algo = static_cast<gamax::groupby::groupby_algo>(algo_int);
    return h;
}

std::optional<gamax::join::join_hint>
parse_join_axis(const std::string& v) {
    if (v == kNA) return std::nullopt;
    // Decoupled from `tpch_params::parse_algo_value` because the
    // legacy map omits the HASH+GFTR pair (the default our IR path
    // ships with). Spell the (algo, mat) pair explicitly here.
    using gamax::join::join_algo;
    using gamax::join::materialization_algo;
    static const std::unordered_map<std::string,
        std::pair<join_algo, materialization_algo>> table = {
        {"hash_gfur",            {gamax::join::HASH,      gamax::join::GFUR}},
        {"hash_gftr",            {gamax::join::HASH,      gamax::join::GFTR}},
        {"sort_merge_gfur",      {gamax::join::SORT,      gamax::join::GFUR}},
        {"sort_merge_gftr",      {gamax::join::SORT,      gamax::join::GFTR}},
        {"partition_hash_gfur",  {gamax::join::PARTITION, gamax::join::GFUR}},
        {"partition_hash_gftr",  {gamax::join::PARTITION, gamax::join::GFTR}},
    };
    auto it = table.find(v);
    if (it == table.end()) {
        throw std::runtime_error("Unknown join axis value: " + v);
    }
    gamax::join::join_hint h;
    h.join_algo = it->second.first;
    h.mat_algo  = it->second.second;
    return h;
}

// ── Plan inspection for skip-by-n/a ───────────────────────────────
//
// Compute the StepKind set of a lowered plan so the fixture can
// decide skip-or-set per axis. Reorder, ScalarAgg, Distinct, Concat,
// CrossJoin, SourceTable have no v1 axis — they're not consulted.

std::unordered_set<int> step_kind_set(const gamax::exec::ExecPlan& ep) {
    std::unordered_set<int> kinds;
    for (const auto& s : ep.steps) {
        kinds.insert(static_cast<int>(s.kind));
    }
    return kinds;
}

bool plan_has(const std::unordered_set<int>& kinds, gamax::exec::StepKind k) {
    return kinds.count(static_cast<int>(k)) > 0;
}

gamax::ir::PlanPtr build_query(int qid, const gamax::ir::QueryBuilder& qb) {
    using namespace gamax::ir::tpch;
    switch (qid) {
        case 1:  return build_q1(qb);
        case 2:  return build_q2(qb);
        case 3:  return build_q3(qb);
        case 4:  return build_q4(qb);
        case 5:  return build_q5(qb);
        case 6:  return build_q6(qb);
        case 7:  return build_q7(qb);
        case 8:  return build_q8(qb);
        case 9:  return build_q9(qb);
        case 10: return build_q10(qb);
        case 11: return build_q11(qb);
        case 12: return build_q12(qb);
        case 13: return build_q13(qb);
        case 14: return build_q14(qb);
        case 15: return build_q15(qb);
        case 16: return build_q16(qb);
        case 17: return build_q17(qb);
        case 18: return build_q18(qb);
        case 19: return build_q19(qb);
        case 20: return build_q20(qb);
        case 21: return build_q21(qb);
        case 22: return build_q22(qb);
        default:
            throw std::runtime_error(
                "tpch_ir_nvbench: unknown query " + std::to_string(qid));
    }
}

void bench_tpch_ir(nvbench::state& state) {
    const int qid = static_cast<int>(state.get_int64("query"));
    const int sf  = static_cast<int>(state.get_int64("scale"));

    // Per-cell pool reset. UserSpaceMM never returns blocks to the
    // driver on its own, so without this every sweep would carry a
    // monotonically-growing freelist + fragmentation across all 22 ×
    // {scale} cells. release_all_mem() does NOT touch the reader's
    // pinned-host cache (it's host memory, not GPU), so column-decode
    // amortization across cells is preserved. Sample-to-sample within
    // this cell still benefits from the post-reset freelist warming
    // up over the first few samples.
    release_all_mem();

    // ── Read operator-kind axes (each accepts an "n/a" sentinel) ──
    auto filter_hint   = parse_filter_axis(state.get_string("filter"));
    auto project_hint  = parse_project_axis(state.get_string("project"));
    auto sort_hint     = parse_sort_axis(state.get_string("sort"));
    auto group_by_hint = parse_group_by_axis(state.get_string("group_by"));
    auto join_hint     = parse_join_axis(state.get_string("join"));

    // Build the IR plan once per cell. The plan / catalog / resolver
    // outlive the lambda; NVBench reuses them across all samples of
    // this cell. Per-cell scope means the resolver's device-table
    // cache is freed when this function returns, bounding GPU memory.
    auto& catalog = shared_catalog();
    gamax::ir::IdGenerator gen;
    gamax::ir::QueryBuilder qb(catalog, gen);
    auto plan = build_query(qid, qb);
    plan = gamax::ir::apply_rules(plan);

    // First lowering with default options — used only to inspect the
    // plan's StepKind set so we can decide skip-by-n/a per axis. If
    // the cell is valid, we re-lower with the populated overrides.
    auto inspect_ep = gamax::ir::lower_to_exec_plan(plan, catalog);
    const auto kinds = step_kind_set(inspect_ep);

    // Skip-by-n/a: a cell is invalid if (axis="n/a" AND its kind is
    // present) OR (axis≠"n/a" AND its kind is absent). The "sort"
    // axis covers BOTH Sort and TopK steps (they share the same
    // sort_table_options hint type).
    using gamax::exec::StepKind;
    const bool has_filter   = plan_has(kinds, StepKind::Filter);
    const bool has_project  = plan_has(kinds, StepKind::Project);
    const bool has_sort     = plan_has(kinds, StepKind::Sort)
                           || plan_has(kinds, StepKind::TopK);
    const bool has_group_by = plan_has(kinds, StepKind::GroupBy);
    const bool has_join     = plan_has(kinds, StepKind::EqJoin)
                           || plan_has(kinds, StepKind::SemiJoin)
                           || plan_has(kinds, StepKind::AntiJoin);

    auto axis_mismatch = [&](bool axis_set, bool kind_present, const char* name) {
        if (axis_set && !kind_present) {
            state.skip(std::string{"axis "} + name +
                       " set but plan has no " + name + " step");
            return true;
        }
        if (!axis_set && kind_present) {
            state.skip(std::string{"axis "} + name +
                       " is n/a but plan has " + name + " step(s)");
            return true;
        }
        return false;
    };
    if (axis_mismatch(filter_hint.has_value(),   has_filter,   "filter"))   return;
    if (axis_mismatch(project_hint.has_value(),  has_project,  "project"))  return;
    if (axis_mismatch(sort_hint.has_value(),     has_sort,     "sort"))     return;
    if (axis_mismatch(group_by_hint.has_value(), has_group_by, "group_by")) return;
    if (axis_mismatch(join_hint.has_value(),     has_join,     "join"))     return;

    // Re-lower with the harness's per-kind overrides. Empty optionals
    // fall through to the engine default.
    gamax::ir::LowerOptions opts;
    opts.filter    = filter_hint;
    opts.project   = project_hint;
    opts.sort      = sort_hint;
    opts.top_k     = sort_hint;     // share with Sort axis
    opts.group_by  = group_by_hint;
    opts.eq_join   = join_hint;     // unified join axis
    opts.semi_join = join_hint;
    opts.anti_join = join_hint;
    auto ep = gamax::ir::lower_to_exec_plan(plan, catalog, opts);

    auto& reader = shared_reader(sf);
    gamax::tpch::IrSourceResolver resolver(reader, catalog);
    gamax::exec::SourceResolver src_fn =
        [&resolver](const std::string& name,
                    const std::vector<std::string>& cols) {
            return resolver(name, cols);
        };

    // Optional per-step breakdown sink. Enabled by setting the
    // GAMAX_BENCH_STEP_CSV env var to a target path. NVBench doesn't
    // accept user-defined CLI flags, so we use an env var. When unset,
    // the sink stays null and ctx.profile_sink is null — zero overhead
    // on the timed path.
    const char* step_csv_path = std::getenv("GAMAX_BENCH_STEP_CSV");
    std::unique_ptr<gamax::exec::StepProfileSink> sink;
    if (step_csv_path && *step_csv_path) {
        sink = std::make_unique<gamax::exec::StepProfileSink>();
    }

    // NVBench's `exec_tag::timer` mode: we control timer.start/stop.
    // `nvbench::exec_tag::sync` ensures NVBench syncs the stream
    // around each sample so cudaEvents-on-stream timing is accurate.
    //
    // The first sample carries JIT and one-time costs (NVBench's hot
    // mode runs a dedicated warmup; cold mode does not, so we drop
    // the first sample ourselves to keep the aggregator's mean/stddev
    // representative of the steady state).
    bool warmed = false;
    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
        [&](nvbench::launch& launch, auto& timer) {
            gamax::phys::Context ctx{launch.get_stream(), sink.get()};
            // Phase 1: resolve sources, sync stream. Untimed.
            auto resolved = gamax::exec::bench_resolve_sources(
                ep, src_fn, launch.get_stream());
            // Phase 2: run the rest of the plan. Timed.
            timer.start();
            auto result = gamax::exec::bench_run_steps(
                ep, std::move(resolved), ctx);
            timer.stop();
            // Drain this sample's per-step events. Skip the first
            // sample (JIT warmup) so subsequent stats are clean.
            if (sink) {
                if (warmed) {
                    sink->ingest_sample(launch.get_stream());
                } else {
                    sink->discard_pending(launch.get_stream());
                    warmed = true;
                }
            }
            // `result` falls out of scope at end of sample; the memory
            // pool reclaims its buffers.
        });

    // Per-cell append to the per-step CSV. Header is written only when
    // the file is new or empty so a multi-cell sweep produces one
    // tidy file. The check races with concurrent writers; this binary
    // is single-process so that's acceptable.
    if (sink && step_csv_path) {
        bool write_header = true;
        {
            std::ifstream probe(step_csv_path);
            if (probe.good() && probe.peek() != std::ifstream::traits_type::eof()) {
                write_header = false;
            }
        }
        std::ofstream out(step_csv_path, std::ios::app);
        if (out) {
            sink->to_summary_csv(out, std::to_string(qid),
                                 static_cast<int64_t>(sf), write_header);
        }
    }
}

} // anonymous namespace

NVBENCH_BENCH(bench_tpch_ir)
    .set_name("tpch_ir")
    .add_int64_axis("query",
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22})
    .add_int64_axis("scale", {1, 10})
    .add_string_axis("filter",   {"fuse", "nofuse",       "n/a"})
    .add_string_axis("project",  {"fuse", "nofuse",       "n/a"})
    .add_string_axis("sort",     {"skf",  "no_skf",       "n/a"})
    .add_string_axis("group_by", {"hash", "sort_gftr",    "sort_gftr_opt",
                                  "sort_gfur", "partition_gftr",
                                  "partition_gfur",       "n/a"})
    .add_string_axis("join",     {"hash_gfur",            "hash_gftr",
                                  "sort_merge_gfur",      "sort_merge_gftr",
                                  "partition_hash_gfur",  "partition_hash_gftr",
                                  "n/a"});
