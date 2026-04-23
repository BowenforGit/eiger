// Small end-to-end runner for the IR → ExecPlan → execute pipeline
// against live TPC-H Parquet data.
//
// Usage:
//   run_tpch_ir <query-id> [--scale N] [--print-plan]
//
// Example:
//   TPCH_DATA_PATH=/scratch/wubo ./run_tpch_ir 6 --scale 1
//
// Only queries whose physical path is wired today are implemented
// (Q6 is the canonical one). Others will exit with a clear message.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "exec/executor.hpp"
#include "exec/print.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "ir/print.hpp"
#include "ir/rules.hpp"
#include "utils/debug_print.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "phys/phys.hpp"
#include "tpch/ir_source_resolver.hpp"
#include "tpch/tpch.hpp"
#include "tpch/tpch_ir.hpp"

namespace {

struct Args {
    int query_id {6};
    int scale_factor {1};
    bool print_plan {false};
};

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc >= 2) a.query_id = std::atoi(argv[1]);
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            a.scale_factor = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--print-plan") == 0) {
            a.print_plan = true;
        } else {
            throw std::runtime_error("Unknown arg: " + std::string(argv[i]));
        }
    }
    return a;
}

gamax::ir::PlanPtr build_query(
    int qid, const gamax::ir::QueryBuilder& qb)
{
    using namespace gamax::ir::tpch;
    switch (qid) {
        case 1:  return build_q1(qb);
        case 2:  return build_q2(qb);
        case 6:  return build_q6(qb);
        case 11: return build_q11(qb);
        case 14: return build_q14(qb);
        case 15: return build_q15(qb);
        case 18: return build_q18(qb);
        case 20: return build_q20(qb);
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

int main(int argc, char** argv) try {
    auto args = parse_args(argc, argv);

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
    auto ep = gamax::ir::lower_to_exec_plan(plan, catalog);
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
    auto t0 = std::chrono::steady_clock::now();
    gamax::exec::SourceResolver source_fn =
        [&resolver](const std::string& name,
                    const std::vector<std::string>& cols) {
            return resolver(name, cols);
        };
    auto result = gamax::exec::execute(ep, source_fn, ctx);
    cudaStreamSynchronize(params.stream);
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Executed in " << elapsed_ms << " ms.\n";

    // ── 5. Print ─────────────────────────────────────────────────
    print_result(result->view(), /*max_rows=*/20);
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
