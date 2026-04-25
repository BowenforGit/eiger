// Small end-to-end runner for the IR → ExecPlan → execute pipeline
// against live TPC-H Parquet data.
//
// Usage:
//   run_tpch_ir <query-id> [--scale N] [--print-plan] [--verify]
//
// Example:
//   TPCH_DATA_PATH=/scratch/wubo ./run_tpch_ir 6 --scale 1 --verify
//
// --verify loads the Parquet golden at
//   $TPCH_DATA_PATH/sf<N>/golden/q<NN>.parquet
// (produced by `script/gen_tpch_golden.py`) and compares column-by-
// column against the gamax output. Prints "VERIFY: OK" on match or a
// diagnostic + exits 1 on mismatch.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <nvtx3/nvtx3.hpp>

#include "exec/executor.hpp"
#include "exec/print.hpp"
#include "exec/step_profile.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "ir/print.hpp"
#include "ir/rules.hpp"
#include "utils/debug_print.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "memory/table.hpp"
#include "phys/phys.hpp"
#include "tpch/ir_source_resolver.hpp"
#include "tpch/tpch.hpp"
#include "tpch/tpch_ir.hpp"
#include "tpch/verify.hpp"

namespace {

struct Args {
    int query_id {6};
    int scale_factor {1};
    bool print_plan {false};
    bool verify {false};
    bool no_fuse_project {false};
    bool use_cuda_mm {false};
    std::string eq_join_algo;  // empty = default
    std::string profile_csv;   // empty = disabled
};

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc >= 2) a.query_id = std::atoi(argv[1]);
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            a.scale_factor = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--print-plan") == 0) {
            a.print_plan = true;
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            a.verify = true;
        } else if (std::strcmp(argv[i], "--profile-csv") == 0 && i + 1 < argc) {
            a.profile_csv = argv[++i];
        } else if (std::strcmp(argv[i], "--no-fuse-project") == 0) {
            a.no_fuse_project = true;
        } else if (std::strcmp(argv[i], "--use-cuda-mm") == 0) {
            // Switches the global memory manager from UserSpaceMM
            // (88 GiB pre-alloc'd slab) to CudaMM (per-allocation
            // cudaMallocAsync). Useful under compute-sanitizer:
            // each CUDA allocation is a separately-tracked range, so
            // an OOB write inside what UserSpaceMM would consider
            // "the slab" is now caught by the sanitizer.
            a.use_cuda_mm = true;
        } else if (std::strcmp(argv[i], "--eq-join-algo") == 0 && i + 1 < argc) {
            a.eq_join_algo = argv[++i];
        } else {
            throw std::runtime_error("Unknown arg: " + std::string(argv[i]));
        }
    }
    return a;
}

std::string golden_path_for(int qid, int scale_factor) {
    const char* data_path = std::getenv("TPCH_DATA_PATH");
    if (!data_path || !*data_path) {
        throw std::runtime_error(
            "--verify requires TPCH_DATA_PATH to be set (golden lives at "
            "$TPCH_DATA_PATH/sf<N>/golden/q<NN>.parquet)");
    }
    std::ostringstream os;
    os << data_path << "/sf" << scale_factor << "/golden/q"
       << (qid < 10 ? "0" : "") << qid << ".parquet";
    return os.str();
}

gamax::ir::PlanPtr build_query(
    int qid, const gamax::ir::QueryBuilder& qb)
{
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
                "query " + std::to_string(qid) + " not wired in this runner.");
    }
}

// Print the first few rows of each column. Types we support here:
// INT32, INT64, FLOAT64, STRING — the set actually produced by the
// TPC-H queries we can run.
void print_result(const gamax::table_view& v, gamax::size_type max_rows) {
    using namespace gamax;
    gamax::size_type n = std::min(max_rows, v.num_rows());
    std::cout << "Result: " << v.num_rows() << " rows × "
              << v.num_columns() << " columns (showing " << n << ")\n";

    // Gather each column to host, dispatch by type.
    std::vector<std::vector<std::string>> cells(v.num_columns());
    for (int c = 0; c < v.num_columns(); ++c) {
        auto col = v[c];
        auto id = col.type().id();
        std::vector<std::string> col_cells;
        col_cells.reserve(n);
        switch (id) {
            case type_id::INT32: {
                auto h = copy_column_to_host<int32_t>(col);
                for (gamax::size_type i = 0; i < n; ++i)
                    col_cells.push_back(std::to_string(h[i]));
                break;
            }
            case type_id::INT64: {
                auto h = copy_column_to_host<int64_t>(col);
                for (gamax::size_type i = 0; i < n; ++i)
                    col_cells.push_back(std::to_string(h[i]));
                break;
            }
            case type_id::FLOAT64: {
                auto h = copy_column_to_host<double>(col);
                for (gamax::size_type i = 0; i < n; ++i)
                    col_cells.push_back(std::to_string(h[i]));
                break;
            }
            case type_id::STRING: {
                // Arrow string columns carry a 2-entry helper buffer
                // of [data_ptr, meta_ptr]. Reading both via one D2H
                // copy keeps the path simple and side-steps the NVCC
                // void*-arithmetic warning in `string_column_view`'s
                // host ctor (harmless in isolation, but grep hits it
                // every time this TU recompiles). Offsets in
                // meta_buffer are INT64 (see get_arrow_string_offsets).
                void* host_ptrs[2] = {nullptr, nullptr};
                cudaMemcpy(host_ptrs, col.data<void>(),
                           2 * sizeof(void*), cudaMemcpyDeviceToHost);
                auto num = col.size();
                std::vector<gamax::size_type> offsets(num + 1);
                cudaMemcpy(offsets.data(), host_ptrs[1],
                           (num + 1) * sizeof(gamax::size_type),
                           cudaMemcpyDeviceToHost);
                gamax::size_type total = offsets.back();
                std::string bytes(total, '\0');
                if (total > 0) {
                    cudaMemcpy(bytes.data(), host_ptrs[0],
                               total, cudaMemcpyDeviceToHost);
                }
                for (gamax::size_type i = 0; i < n; ++i) {
                    col_cells.emplace_back(
                        bytes.data() + offsets[i],
                        offsets[i + 1] - offsets[i]);
                }
                break;
            }
            default:
                for (gamax::size_type i = 0; i < n; ++i) {
                    col_cells.push_back("<" +
                        std::to_string(static_cast<int>(id)) + ">");
                }
                break;
        }
        cells[c] = std::move(col_cells);
    }
    for (gamax::size_type i = 0; i < n; ++i) {
        for (int c = 0; c < v.num_columns(); ++c) {
            if (c) std::cout << " | ";
            std::cout << cells[c][i];
        }
        std::cout << "\n";
    }
}

} // namespace

// Map an `--eq-join-algo <name>` value to a `gamax::join::join_hint`.
// Same value vocabulary as the bench fixture's `eq_join` axis.
std::optional<gamax::join::join_hint> parse_eq_join_algo(const std::string& v) {
    if (v.empty()) return std::nullopt;
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
        throw std::runtime_error("Unknown --eq-join-algo: " + v);
    }
    gamax::join::join_hint h;
    h.join_algo = it->second.first;
    h.mat_algo  = it->second.second;
    return h;
}

int main(int argc, char** argv) try {
    auto args = parse_args(argc, argv);

    // `use_cuda_mm()` must run before any get_mm() — wire it up here
    // before any TPC-H reader / executor work touches device memory.
    if (args.use_cuda_mm) {
        use_cuda_mm();
        std::cout << "Memory manager: CudaMM (cudaMallocAsync per allocation)\n";
    }

    std::cout << "== TPC-H Q" << args.query_id
              << " (SF=" << args.scale_factor << ") via IR pipeline ==\n";

    // ── 1. Build catalog + IR ────────────────────────────────────
    auto catalog = gamax::ir::tpch::make_tpch_catalog();
    gamax::ir::IdGenerator gen;
    gamax::ir::QueryBuilder qb(catalog, gen);
    auto plan = build_query(args.query_id, qb);
    auto pre_rules_plan = plan;  // keep a reference for the plan dump
    std::cout << "IR built: " << plan->output_schema.size()
              << " output columns.\n";
    if (args.print_plan) {
        std::cout << "\n-- IR (pre-rules) --\n"
                  << gamax::ir::format_plan(*plan) << "\n";
    }

    // ── 2. Apply IR rewrite rules (defaults-on) ──────────────────
    plan = gamax::ir::apply_rules(plan);
    if (args.print_plan) {
        std::cout << "\n-- IR (post-rules) --\n"
                  << gamax::ir::format_plan(*plan) << "\n";
    }

    // ── 3. Lower to ExecPlan ─────────────────────────────────────
    gamax::ir::LowerOptions lower_opts;
    if (args.no_fuse_project) {
        gamax::project::project_options po;
        po.fuse_operations = false;
        lower_opts.project = po;
    }
    if (auto h = parse_eq_join_algo(args.eq_join_algo)) {
        lower_opts.eq_join = h;
    }
    auto ep = gamax::ir::lower_to_exec_plan(plan, catalog, lower_opts);
    std::cout << "ExecPlan: " << ep.steps.size() << " steps, "
              << ep.required_sources.size() << " sources.\n";
    if (args.print_plan) {
        std::cout << "\n-- ExecPlan --\n"
                  << gamax::exec::format_exec_plan(ep) << "\n";
    }

    // Env-var-driven plan dump to stderr. Self-gated: silently does
    // nothing unless `GAMAX_PHYS_DEBUG` is non-empty or
    // `GAMAX_DEBUG_PLAN=1`. Runs in parallel with the --print-plan flag
    // path above, so you can use either / both.
    gamax::util::debug::print_plans_if_debug(*pre_rules_plan, *plan, ep);

    // ── 3. Load sources via the TPC-H reader ─────────────────────
    gamax::tpch::tpch_params params(args.scale_factor);
    gamax::tpch::TpchTableReader reader(params);
    gamax::tpch::IrSourceResolver resolver(reader, catalog);

    // ── 4. Execute ───────────────────────────────────────────────
    gamax::phys::Context ctx{params.stream};

    // Opt-in per-step GPU timing. The sink is attached to `ctx` so the
    // executor's step walker can record cudaEvent pairs around each
    // phys dispatch; we finalize/dump after execute().
    std::unique_ptr<gamax::exec::StepProfileSink> profile_sink;
    if (!args.profile_csv.empty()) {
        profile_sink = std::make_unique<gamax::exec::StepProfileSink>();
        ctx.profile_sink = profile_sink.get();
    }

    // Per-query NVTX range: groups the per-step ranges under one
    // parent bar in the Nsight Systems timeline. Label includes the
    // query id and scale factor so sweeps are easy to compare visually.
    std::string query_label =
        "Q" + std::to_string(args.query_id) +
        "_SF" + std::to_string(args.scale_factor);
    nvtxRangePushA(query_label.c_str());

    gamax::exec::SourceResolver source_fn =
        [&resolver](const std::string& name,
                    const std::vector<std::string>& cols) {
            return resolver(name, cols);
        };
    auto t0 = std::chrono::steady_clock::now();
    auto result = gamax::exec::execute(ep, source_fn, ctx);
    cudaStreamSynchronize(params.stream);
    auto t1 = std::chrono::steady_clock::now();
    nvtxRangePop();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Executed in " << elapsed_ms << " ms.\n";

    // Resolve per-step GPU timings (if enabled) and append to the CSV.
    // Header is only written when the file is new or empty, so a sweep
    // over 22 queries produces one well-formed CSV.
    if (profile_sink) {
        profile_sink->finalize(params.stream);
        bool write_header = true;
        {
            std::ifstream probe(args.profile_csv);
            if (probe.good() && probe.peek() != std::ifstream::traits_type::eof()) {
                write_header = false;
            }
        }
        std::ofstream out(args.profile_csv, std::ios::app);
        if (!out) {
            throw std::runtime_error(
                "Failed to open profile CSV for append: " + args.profile_csv);
        }
        profile_sink->to_csv(out, query_label, write_header);
        std::cout << "PROFILE: appended " << profile_sink->rows().size()
                  << " step rows to " << args.profile_csv << "\n";
    }

    // ── 5. Print ─────────────────────────────────────────────────
    print_result(result->view(), /*max_rows=*/20);

    // ── 6. Optional verify against Parquet golden ────────────────
    if (args.verify) {
        auto golden = golden_path_for(args.query_id, args.scale_factor);
        std::cout << "VERIFY: comparing against " << golden << "\n";
        auto vr = gamax::tpch::verify_result(result->view(), golden);
        if (vr.ok) {
            std::cout << "VERIFY: OK\n";
        } else {
            std::cout << "VERIFY: MISMATCH\n" << vr.diagnostic;
            return 1;
        }
    }
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
