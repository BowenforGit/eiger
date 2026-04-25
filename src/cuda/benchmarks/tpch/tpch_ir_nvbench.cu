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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "exec/executor.hpp"
#include "exec/step_profile.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "ir/rules.hpp"
#include "memory/mem_manager.hpp"
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

    // Build the IR plan once per cell. The plan / catalog / resolver
    // outlive the lambda; NVBench reuses them across all samples of
    // this cell. Per-cell scope means the resolver's device-table
    // cache is freed when this function returns, bounding GPU memory.
    auto& catalog = shared_catalog();
    gamax::ir::IdGenerator gen;
    gamax::ir::QueryBuilder qb(catalog, gen);
    auto plan = build_query(qid, qb);
    plan = gamax::ir::apply_rules(plan);
    auto ep = gamax::ir::lower_to_exec_plan(plan, catalog);

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
    .add_int64_axis("scale", {1, 10});
