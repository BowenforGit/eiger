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
//   ./benchmark_tpch_ir                            # full breakdown sweep
//   ./benchmark_tpch_ir -a query=6 -a scale=10     # one query, both scales
//   ./benchmark_tpch_ir --profile -a query=6       # one-shot, no timing
//                                                  # (attach `nsys profile`)
//   ./benchmark_tpch_ir -a query=3 -a scale=1      # breakdown for Q3:
//                                                  # 1 baseline + per-axis algo
//                                                  # cells, multi-algo cells
//                                                  # are skipped automatically
//
// Operator axes (each accepts "default" + "n/a" sentinels). The
// concrete value that the bench's "default" sentinel resolves to is
// intentionally NOT listed as a separate axis value — running e.g.
// "filter=nofuse" alongside "filter=default" would yield two cells
// with the same physical config (both fuse_operations=false). The
// cells listed below are exactly the *non-default* algo choices.
//   filter, project   "fuse"   "default"   "n/a"            (default = nofuse)
//   sort              "skf"    "default"   "n/a"            (default = no_skf)
//   group_by          "sort_gftr"   "sort_gftr_opt"   "sort_gfur"
//                     "partition_gftr"   "partition_gfur"
//                     "default"   "n/a"                     (default = hash)
//   distinct          "hash_skf"   "sort"   "sort_skf"
//                     "default"   "n/a"                     (default = hash, no SKF)
//   eq_join           "sort_merge_gfur"   "sort_merge_gftr"
//                     "partition_hash_gfur"   "partition_hash_gftr"
//                     "default"   "n/a"                     (default = hash_gfur)
//
// SemiJoin and AntiJoin are distinct StepKinds with only one
// implementation each today, so they have no axis. Their payload
// hints are populated by lowering with `default_eq_join_hint()` (the
// same fallback ad-hoc callers rely on).
//
// Three axis modes per operator:
//   "n/a"     — kind is absent from the plan; cell is skipped if the
//               kind is in fact present.
//   "default" — kind is present; lowering uses the bench-local default
//               (see `bench_defaults` below — explicit so the baseline
//               doesn't drift with IR-engine refactors). Cell is
//               skipped if the kind is absent.
//   <algo>    — kind is present and lowering applies this hint. Cell
//               is skipped if the kind is absent.
//
// Breakdown-only constraint: the harness skips any cell where two or
// more axes are in algo (non-default, non-n/a) mode. The rationale is
// that operators in a query are independent — sweeping their hints
// jointly produces highly redundant cells for break-down analysis.
// Each query produces one all-default baseline plus, for each axis
// whose kind is present, one cell per algo value. A query with all 5
// kinds yields 1 + 1 + 1 + 1 + 5 + 4 = 13 cells per scale.
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
// Each operator-kind axis is a string axis with three modes:
//   "n/a"      — kind absent from plan
//   "default"  — kind present, use bench-local default (see
//                `bench_defaults` below)
//   <algo>     — kind present, override with this hint
//
// `parse_*_axis` helpers are only called in algo mode — callers must
// pre-check `is_set(v)` before invoking them.

constexpr const char* kNA = "n/a";
constexpr const char* kDefault = "default";

bool is_na(const std::string& v)      { return v == kNA; }
bool is_default(const std::string& v) { return v == kDefault; }
bool is_set(const std::string& v)     { return !is_na(v) && !is_default(v); }

// Bench-local defaults — explicit so the baseline cell is stable across
// IR-engine refactors (the engine's own defaults are deliberately
// hidden from this binary). The "default" axis mode applies these.
namespace bench_defaults {
gamax::select::select_options filter() {
    gamax::select::select_options o;
    o.fuse_operations = false;        // nofuse
    return o;
}
gamax::project::project_options project() {
    gamax::project::project_options o;
    o.fuse_operations = false;        // nofuse
    return o;
}
gamax::sort::sort_table_options sort() {
    gamax::sort::sort_table_options o;
    o.smart_keys_fusion = false;      // no_skf
    return o;
}
gamax::groupby::groupby_hint group_by() {
    gamax::groupby::groupby_hint h;
    h.groupby_algo = gamax::groupby::HASH;
    return h;
}
gamax::groupby::groupby_hint distinct() {
    gamax::groupby::groupby_hint h;
    h.groupby_algo = gamax::groupby::HASH;
    h.enable_skf   = false;
    return h;
}
gamax::join::join_hint eq_join() {
    gamax::join::join_hint h;
    h.join_algo = gamax::join::HASH;
    h.mat_algo  = gamax::join::GFUR;
    return h;
}
} // namespace bench_defaults

gamax::select::select_options
parse_filter_axis(const std::string& v) {
    gamax::select::select_options o;
    o.fuse_operations = (v == "fuse");
    return o;
}

gamax::project::project_options
parse_project_axis(const std::string& v) {
    gamax::project::project_options o;
    o.fuse_operations = (v == "fuse");
    return o;
}

gamax::sort::sort_table_options
parse_sort_axis(const std::string& v) {
    gamax::sort::sort_table_options o;
    o.smart_keys_fusion = (v == "skf");
    return o;
}

gamax::groupby::groupby_hint
parse_group_by_axis(const std::string& v) {
    int algo_int = gamax::tpch::tpch_params::parse_algo_value(v);
    gamax::groupby::groupby_hint h;
    h.groupby_algo = static_cast<gamax::groupby::groupby_algo>(algo_int);
    return h;
}

// Distinct axis. Decoupled from `parse_algo_value` because Distinct's
// supported space is a strict subset of groupby's: HASH (no SKF) and
// HASH+SKF, SORT (no SKF) and SORT+SKF. PARTITION variants are
// intentionally absent — `phys::distinct` has no partition impl and
// throws if asked. Spell the four (algo, skf) pairs explicitly.
gamax::groupby::groupby_hint
parse_distinct_axis(const std::string& v) {
    gamax::groupby::groupby_hint h;
    if (v == "hash_skf") {
        h.groupby_algo = gamax::groupby::HASH;
        h.enable_skf   = true;
    } else if (v == "sort") {
        h.groupby_algo = gamax::groupby::SORT;
        h.enable_skf   = false;
    } else if (v == "sort_skf") {
        h.groupby_algo = gamax::groupby::SORT;
        h.enable_skf   = true;
    } else {
        throw std::runtime_error("Unknown distinct axis value: " + v);
    }
    return h;
}

// `eq_join` axis only — SemiJoin / AntiJoin have one implementation
// each today, so the harness doesn't expose an axis for them.
gamax::join::join_hint
parse_eq_join_axis(const std::string& v) {
    // Decoupled from `tpch_params::parse_algo_value` because the
    // legacy map omits the HASH+GFTR pair (the default our IR path
    // ships with). Spell the (algo, mat) pair explicitly here.
    using gamax::join::join_algo;
    using gamax::join::materialization_algo;
    // HASH+GFTR is intentionally absent — it is not a supported
    // (algo, mat) combination on the inner_eq_join path.
    static const std::unordered_map<std::string,
        std::pair<join_algo, materialization_algo>> table = {
        {"hash_gfur",            {gamax::join::HASH,      gamax::join::GFUR}},
        {"sort_merge_gfur",      {gamax::join::SORT,      gamax::join::GFUR}},
        {"sort_merge_gftr",      {gamax::join::SORT,      gamax::join::GFTR}},
        {"partition_hash_gfur",  {gamax::join::PARTITION, gamax::join::GFUR}},
        {"partition_hash_gftr",  {gamax::join::PARTITION, gamax::join::GFTR}},
    };
    auto it = table.find(v);
    if (it == table.end()) {
        throw std::runtime_error("Unknown eq_join axis value: " + v);
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
        // Alternative Q7 plans (see tpch_ir.cu build_q7_alt).
        case 71: return build_q71(qb);
        case 72: return build_q72(qb);
        case 73: return build_q73(qb);
        default:
            throw std::runtime_error(
                "tpch_ir_nvbench: unknown query " + std::to_string(qid));
    }
}

void bench_tpch_ir(nvbench::state& state) {
    const int qid = static_cast<int>(state.get_int64("query"));
    const int sf  = static_cast<int>(state.get_int64("scale"));

    // ── Read operator-kind axes (raw strings; tri-state per axis) ──
    const auto fil_v = state.get_string("filter");
    const auto prj_v = state.get_string("project");
    const auto srt_v = state.get_string("sort");
    const auto grp_v = state.get_string("group_by");
    const auto dst_v = state.get_string("distinct");
    const auto ejn_v = state.get_string("eq_join");

    // Breakdown-only constraint: at most one axis is in algo (set)
    // mode. Cells with multiple non-default-non-n/a axes are skipped —
    // operators are independent for break-down analysis, so jointly
    // sweeping them produces highly redundant cells. Done BEFORE the
    // per-cell pool reset so the ~99% of cells that this prunes don't
    // pay the slab-realloc cost.
    const int set_count =
        static_cast<int>(is_set(fil_v))
      + static_cast<int>(is_set(prj_v))
      + static_cast<int>(is_set(srt_v))
      + static_cast<int>(is_set(grp_v))
      + static_cast<int>(is_set(dst_v))
      + static_cast<int>(is_set(ejn_v));
    if (set_count > 1) {
        state.skip("more than one axis is non-default — break-down sweep "
                   "varies one operator at a time");
        return;
    }

    // Per-cell pool reset. UserSpaceMM never returns blocks to the
    // driver on its own, so without this every sweep would carry a
    // monotonically-growing freelist + fragmentation across all 22 ×
    // {scale} cells. release_all_mem() does NOT touch the reader's
    // pinned-host cache (it's host memory, not GPU), so column-decode
    // amortization across cells is preserved. Sample-to-sample within
    // this cell still benefits from the post-reset freelist warming
    // up over the first few samples.
    release_all_mem();

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
    // plan's StepKind set so we can decide skip-by-presence per axis.
    // If the cell is valid, we re-lower with the populated overrides.
    auto inspect_ep = gamax::ir::lower_to_exec_plan(plan, catalog);
    const auto kinds = step_kind_set(inspect_ep);

    // Skip-by-presence: an axis in non-NA mode (default or set) requires
    // its kind to be present; an axis in NA mode requires the kind to
    // be absent. The "sort" axis covers BOTH Sort and TopK steps (they
    // share the same sort_table_options hint type).
    using gamax::exec::StepKind;
    const bool has_filter   = plan_has(kinds, StepKind::Filter);
    const bool has_project  = plan_has(kinds, StepKind::Project);
    const bool has_sort     = plan_has(kinds, StepKind::Sort)
                           || plan_has(kinds, StepKind::TopK);
    const bool has_group_by = plan_has(kinds, StepKind::GroupBy);
    const bool has_distinct = plan_has(kinds, StepKind::Distinct);
    const bool has_eq_join  = plan_has(kinds, StepKind::EqJoin);

    auto axis_mismatch = [&](bool kind_required, bool kind_present, const char* name) {
        if (kind_required && !kind_present) {
            state.skip(std::string{"axis "} + name +
                       " set but plan has no " + name + " step");
            return true;
        }
        if (!kind_required && kind_present) {
            state.skip(std::string{"axis "} + name +
                       " is n/a but plan has " + name + " step(s)");
            return true;
        }
        return false;
    };
    if (axis_mismatch(!is_na(fil_v), has_filter,   "filter"))   return;
    if (axis_mismatch(!is_na(prj_v), has_project,  "project"))  return;
    if (axis_mismatch(!is_na(srt_v), has_sort,     "sort"))     return;
    if (axis_mismatch(!is_na(grp_v), has_group_by, "group_by")) return;
    if (axis_mismatch(!is_na(dst_v), has_distinct, "distinct")) return;
    if (axis_mismatch(!is_na(ejn_v), has_eq_join,  "eq_join"))  return;

    // Re-lower with the harness's per-kind overrides. Every non-NA
    // axis populates its LowerOptions field — SET mode applies the
    // parsed hint, DEFAULT mode applies the bench-local default. NA
    // mode leaves the field empty (kind is absent from the plan).
    // SemiJoin and AntiJoin have a single implementation today, so
    // they get no axis — `opts.semi_join` / `opts.anti_join` stay
    // empty and lowering populates them with `default_eq_join_hint()`,
    // matching ad-hoc callers (run_tpch_ir, tests).
    gamax::ir::LowerOptions opts;
    if (!is_na(fil_v))
        opts.filter   = is_set(fil_v) ? parse_filter_axis(fil_v)
                                      : bench_defaults::filter();
    if (!is_na(prj_v))
        opts.project  = is_set(prj_v) ? parse_project_axis(prj_v)
                                      : bench_defaults::project();
    if (!is_na(srt_v)) {
        auto h = is_set(srt_v) ? parse_sort_axis(srt_v)
                               : bench_defaults::sort();
        opts.sort  = h;
        opts.top_k = h;            // share with Sort axis
    }
    if (!is_na(grp_v))
        opts.group_by = is_set(grp_v) ? parse_group_by_axis(grp_v)
                                      : bench_defaults::group_by();
    if (!is_na(dst_v))
        opts.distinct = is_set(dst_v) ? parse_distinct_axis(dst_v)
                                      : bench_defaults::distinct();
    if (!is_na(ejn_v))
        opts.eq_join  = is_set(ejn_v) ? parse_eq_join_axis(ejn_v)
                                      : bench_defaults::eq_join();
    gamax::ir::tpch::apply_query_hints(qid, opts);
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

    // Pre-flight: run the plan once outside the measurement loop so a
    // throwing cell (multi-key/STRING groupby, partition_hash_join
    // bucket overflow, …) becomes a clean skip rather than tainting
    // NVBench's sample stats or leaving the stream in a bad state.
    //
    // Skipped under GAMAX_BENCH_SKIP_PREFLIGHT=1. Use this when running
    // under `nsys profile` — pre-flight would otherwise execute every
    // step's NVTX range twice (once here, once in state.exec), so
    // nvtx_kern_sum's Total Time would be 2× the actual per-execution
    // value. Set the env var, accept that a throwing cell will fail
    // ungracefully, and post-process accordingly.
    static const bool skip_preflight =
        std::getenv("GAMAX_BENCH_SKIP_PREFLIGHT") != nullptr;
    if (!skip_preflight) {
        try {
            gamax::phys::Context ctx_pre{cudaStreamDefault, /*sink=*/nullptr};
            auto resolved_pre = gamax::exec::bench_resolve_sources(
                ep, src_fn, cudaStreamDefault);
            auto result_pre = gamax::exec::bench_run_steps(
                ep, std::move(resolved_pre), ctx_pre);
            cudaStreamSynchronize(cudaStreamDefault);
            (void)result_pre;
        } catch (const std::exception& e) {
            state.skip(std::string{"execute throws: "} + e.what());
            return;
        }
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
    // tidy file. Includes the per-axis labels so a downstream plot
    // can attribute each step row to a unique cell.
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
            if (write_header) {
                out << "query,scale,filter,project,sort,group_by,distinct,"
                       "eq_join,step_index,step_kind,n_samples,"
                       "mean_ms,min_ms,max_ms,stddev_ms\n";
            }
            for (const auto& row : sink->summary()) {
                out << qid << ',' << sf << ','
                    << fil_v << ',' << prj_v << ',' << srt_v << ','
                    << grp_v << ',' << dst_v << ',' << ejn_v << ','
                    << row.step_index << ',' << row.kind_name << ','
                    << row.n_samples << ','
                    << row.mean_ms << ',' << row.min_ms << ','
                    << row.max_ms  << ',' << row.stddev_ms << '\n';
            }
        }
    }
}

} // anonymous namespace

NVBENCH_BENCH(bench_tpch_ir)
    .set_name("tpch_ir")
    .add_int64_axis("query",
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
         // Alternative Q7 plans (build/probe-convention variants).
         71, 72, 73})
    .add_int64_axis("scale", {1, 10, 30, 100})
    // The bench-default value of each axis is intentionally NOT
    // listed as an explicit choice — see the file header comment for
    // why ("default" + "<the same value>" would produce duplicate
    // cells). Defaults: filter/project=nofuse, sort=no_skf,
    // group_by=hash, distinct=hash_no_skf, eq_join=hash_gfur.
    .add_string_axis("filter",   {"fuse", "default", "n/a"})
    .add_string_axis("project",  {"fuse", "default", "n/a"})
    .add_string_axis("sort",     {"skf",  "default", "n/a"})
    .add_string_axis("group_by", {"sort_gftr",      "sort_gftr_opt",
                                  "sort_gfur",      "partition_gftr",
                                  "partition_gfur", "default", "n/a"})
    .add_string_axis("distinct", {"hash_skf", "sort", "sort_skf",
                                  "default", "n/a"})
    .add_string_axis("eq_join",  {"sort_merge_gfur",     "sort_merge_gftr",
                                  "partition_hash_gfur", "partition_hash_gftr",
                                  "default", "n/a"});
