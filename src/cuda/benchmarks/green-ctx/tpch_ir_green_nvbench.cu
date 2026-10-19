// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Green-context SM-scaling variant of the IR-driven TPC-H benchmark.
//
// This measures how a WHOLE TPC-H query (a DAG of operators) scales as
// the SM budget is shrunk, by routing the entire timed query execution
// onto a CUDA green-context partition of `green_sms` SMs. It is the
// query-level sequel to the per-kernel green-ctx study in this same
// directory (multiset/gather/ieq_join/sort) — a query blends those
// per-operator scaling profiles, so its curve is predicted by which of
// its operators scale (hash probes) vs saturate (gather / phj).
//
// Relationship to tpch_ir_nvbench.cu (the breakdown bench):
//   * We do NOT sweep the operator-algo axes (filter/project/sort/
//     group_by/distinct/eq_join). This bench runs ONE config: the
//     CANONICAL DEFAULT — lowered with an empty LowerOptions +
//     apply_query_hints, exactly like run_tpch_ir's default path. That
//     guarantees the config we TIME here is the same config that
//     `run_tpch_ir --green-sms 152 --verify` validates against the
//     DuckDB golden (the correctness gate for the sweep).
//   * We ADD a `green_sms` int64 axis (0 = native full device). Per
//     cell we create a green ctx spanning ~green_sms SMs and set it as
//     NVBench's stream, so every op the executor launches — source
//     resolution (untimed) and the query steps (timed) — runs on that
//     partition.
//
// Methodology carried over from the breakdown bench:
//   * SourceTable resolution (Parquet decode + H2D) is excluded from
//     the timed region via the bench_resolve_sources / bench_run_steps
//     split.
//   * Process-wide reader + catalog caches amortize decode across cells.
//   * Optional per-step breakdown via GAMAX_BENCH_STEP_CSV — here the
//     per-cell key columns are (query, scale, green_sms) so a downstream
//     plot can show WHICH operator inside a query scales vs saturates.
//
// CLI examples:
//   ./benchmark_tpch_ir_green -a query=6 -a scale=10 \
//       -a "green_sms=[0,8,16,32,48,64,96,128,152]"
//   GAMAX_BENCH_STEP_CSV=steps.csv ./benchmark_tpch_ir_green \
//       -a query=3 -a scale=10 -a green_sms=64
//
// green_sms <= 0  -> native full device (NVBench's default stream).

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

// Green-ctx SM-partition helper. Same directory; provides green_setup /
// green_teardown and the NVBench glue green_bench_stream (reads the
// "green_sms" axis, creates the partition on the current device, records
// an "actual_sms" summary, returns the stream to hand to NVBench).
#include "green_ctx.hpp"

namespace {

// Process-wide reader cache, keyed by scale factor. Holds the per-column
// pinned-host cache + parquet::FileReader handle (host RAM, not GPU), so
// keeping it alive for the full sweep preserves column-decode
// amortization across every cell. Mirrors tpch_ir_nvbench.cu.
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
                "tpch_ir_green_nvbench: unknown query " + std::to_string(qid));
    }
}

void bench_tpch_ir_green(nvbench::state& state) {
    const int qid = static_cast<int>(state.get_int64("query"));
    const int sf  = static_cast<int>(state.get_int64("scale"));

    // Per-cell pool reset. UserSpaceMM never returns blocks to the driver
    // on its own; without this the freelist + fragmentation would grow
    // monotonically across all (query × scale × green_sms) cells.
    // release_all_mem() does NOT touch the reader's pinned-host cache.
    gamax::prewarm_device_memory_resource();

    // Build the IR plan and lower with the CANONICAL DEFAULT config —
    // identical to run_tpch_ir's default path (empty LowerOptions +
    // apply_query_hints). No operator-algo axes here: this bench varies
    // only the SM budget, not the operator algorithm. Matching the
    // runner's lowering is what lets its --verify stand in as the
    // correctness gate for exactly the config timed here.
    auto& catalog = shared_catalog();
    gamax::ir::IdGenerator gen;
    gamax::ir::QueryBuilder qb(catalog, gen);
    auto plan = build_query(qid, qb);
    plan = gamax::ir::apply_rules(plan);

    gamax::ir::LowerOptions opts;
    gamax::ir::tpch::apply_query_hints(qid, opts);
    auto ep = gamax::ir::lower_to_exec_plan(plan, catalog, opts);

    auto& reader = shared_reader(sf);
    gamax::tpch::IrSourceResolver resolver(reader, catalog);
    gamax::exec::SourceResolver src_fn =
        [&resolver](const std::string& name,
                    const std::vector<std::string>& cols) {
            return resolver(name, cols);
        };

    // Optional per-step breakdown sink (env-gated, same side-channel as
    // the breakdown bench). Null → zero overhead on the timed path.
    const char* step_csv_path = std::getenv("GAMAX_BENCH_STEP_CSV");
    std::unique_ptr<gamax::exec::StepProfileSink> sink;
    if (step_csv_path && *step_csv_path) {
        sink = std::make_unique<gamax::exec::StepProfileSink>();
    }

    // Pre-flight: run the plan once (native, default stream) so a
    // throwing cell becomes a clean skip. Skipped under
    // GAMAX_BENCH_SKIP_PREFLIGHT=1 (used under `nsys profile` to avoid
    // double-counting NVTX ranges). NOTE: a stream-0-LEAK crash surfaces
    // only on the green stream, not here — such crashes are ruled out
    // ahead of the sweep by the run_tpch_ir --green-sms --verify gate.
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

    // Create the green-ctx partition for this cell (reads the green_sms
    // axis; green_sms<=0 → returns 0 = native default stream). Then a
    // full-device sync so the untimed pre-flight / default-stream work is
    // drained before the green stream (a SEPARATE context that does not
    // serialize with stream 0) starts. set_cuda_stream is UNCONDITIONAL,
    // matching the per-kernel green benches: native (green_sms=0) times on
    // the default stream (a view of stream 0), so green@152 ≡ native is a
    // clean apples-to-apples check.
    GreenCtx g;
    cudaStream_t gstream = green_bench_stream(state, g);
    cudaDeviceSynchronize();
    state.set_cuda_stream(nvbench::make_cuda_stream_view(gstream));

    bool warmed = false;
    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
        [&](nvbench::launch& launch, auto& timer) {
            gamax::phys::Context ctx{launch.get_stream(), sink.get()};
            // Phase 1: resolve sources on the (green) stream. Untimed.
            auto resolved = gamax::exec::bench_resolve_sources(
                ep, src_fn, launch.get_stream());
            // Phase 2: run the query steps on the (green) stream. Timed.
            timer.start();
            auto result = gamax::exec::bench_run_steps(
                ep, std::move(resolved), ctx);
            timer.stop();
            // Drain this sample's per-step events; skip the first sample
            // (JIT warmup) so the aggregated stats stay representative.
            if (sink) {
                if (warmed) {
                    sink->ingest_sample(launch.get_stream());
                } else {
                    sink->discard_pending(launch.get_stream());
                    warmed = true;
                }
            }
        });

    green_teardown(g);

    // Per-cell append to the per-step CSV. Key columns are (query, scale,
    // green_sms) so each step row attributes to a unique SM-budget cell.
    if (sink && step_csv_path) {
        const int64_t green_sms = state.get_int64_or_default("green_sms", 0);
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
                out << "query,scale,green_sms,step_index,step_kind,"
                       "n_samples,mean_ms,min_ms,max_ms,stddev_ms\n";
            }
            for (const auto& row : sink->summary()) {
                out << qid << ',' << sf << ',' << green_sms << ','
                    << row.step_index << ',' << row.kind_name << ','
                    << row.n_samples << ','
                    << row.mean_ms << ',' << row.min_ms << ','
                    << row.max_ms  << ',' << row.stddev_ms << '\n';
            }
        }
    }
}

} // anonymous namespace

NVBENCH_BENCH(bench_tpch_ir_green)
    .set_name("tpch_ir_green")
    .add_int64_axis("query",
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22})
    .add_int64_axis("scale", {1, 10, 30, 100})
    // 0 = native full device; the rest reuse the per-kernel study's
    // partition sizes so the query curves compare directly to the
    // kernel curves. GB200 = 152 SMs, alignment 8.
    .add_int64_axis("green_sms",
        {0, 8, 16, 32, 48, 64, 96, 128, 152});
