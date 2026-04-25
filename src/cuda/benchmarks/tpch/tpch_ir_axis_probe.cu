// IR-driven TPC-H "axis probe" — cheaper alternative to the NVBench
// sweep when the question is "which (query × axis-config) cells fail
// at all", not "what's the GPU time of cells that succeed".
//
// Bypasses NVBench's measurement loop (per-cell ~1s overhead) and
// instead enumerates the full axis Cartesian itself, calling the IR
// `lower → execute` pipeline once per valid cell with a try/catch.
// Each invalid cell is short-circuited via the same skip-by-n/a logic
// the bench fixture uses; each valid cell either runs to completion
// (PASS) or throws (FAIL: <error>). One CSV row per invalid cell
// would explode the output, so SKIP rows are omitted by default.
//
// Tradeoffs vs benchmark_tpch_ir:
//   * No timing — the goal is failure detection, not perf.
//   * No `release_all_mem` between cells — the pool grows monotonically
//     across the run. SF=1 cells touch a few MB of intermediates each;
//     even 10k cells fit comfortably in the 88 GiB pool. If you want
//     to probe SF=10 or larger, set GAMAX_PROBE_RESET_POOL=1.
//
// Output CSV columns:
//   query,scale,filter,project,sort,group_by,eq_join,status,error
// status is `pass` or `fail`. `error` is empty for pass; for fail it's
// the runtime_error::what() of the throw, with commas+newlines stripped
// for CSV-safety.
//
// CLI:
//   tpch_ir_axis_probe --scale N --out path.csv
//                      [--query-min N] [--query-max N]
//                      [--quiet]   # don't print per-cell progress to stderr
//
// Typical invocation:
//   ./tpch_ir_axis_probe --scale 1 --out /tmp/sf1_probe.csv

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "exec/executor.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "ir/rules.hpp"
#include "memory/mem_manager.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/join/join.hpp"
#include "operators/project/project.hpp"
#include "operators/select/select.hpp"
#include "operators/sort/sort.hpp"
#include "phys/phys.hpp"
#include "tpch/ir_source_resolver.hpp"
#include "tpch/tpch.hpp"
#include "tpch/tpch_ir.hpp"

namespace {

constexpr const char* kNA = "n/a";

// Process-wide reader cache, keyed by scale. Same shape as the
// NVBench fixture — pinned-host caches survive the whole probe run.
gamax::tpch::TpchTableReader& shared_reader(int sf) {
    static std::unordered_map<int,
        std::unique_ptr<gamax::tpch::TpchTableReader>> readers;
    auto it = readers.find(sf);
    if (it == readers.end()) {
        gamax::tpch::tpch_params params(sf);
        auto r = std::make_unique<gamax::tpch::TpchTableReader>(params);
        it = readers.emplace(sf, std::move(r)).first;
    }
    return *it->second;
}

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
                "tpch_ir_axis_probe: unknown query " + std::to_string(qid));
    }
}

// ── Axis parsing (mirrors tpch_ir_nvbench.cu) ─────────────────────

std::optional<gamax::select::select_options>
parse_filter(const std::string& v) {
    if (v == kNA) return std::nullopt;
    gamax::select::select_options o;
    o.fuse_operations = (v == "fuse");
    return o;
}
std::optional<gamax::project::project_options>
parse_project(const std::string& v) {
    if (v == kNA) return std::nullopt;
    gamax::project::project_options o;
    o.fuse_operations = (v == "fuse");
    return o;
}
std::optional<gamax::sort::sort_table_options>
parse_sort(const std::string& v) {
    if (v == kNA) return std::nullopt;
    gamax::sort::sort_table_options o;
    o.smart_keys_fusion = (v == "skf");
    return o;
}
std::optional<gamax::groupby::groupby_hint>
parse_group_by(const std::string& v) {
    if (v == kNA) return std::nullopt;
    int algo = gamax::tpch::tpch_params::parse_algo_value(v);
    gamax::groupby::groupby_hint h;
    h.groupby_algo = static_cast<gamax::groupby::groupby_algo>(algo);
    return h;
}
std::optional<gamax::join::join_hint>
parse_eq_join(const std::string& v) {
    if (v == kNA) return std::nullopt;
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
        throw std::runtime_error("Unknown eq_join axis value: " + v);
    }
    gamax::join::join_hint h;
    h.join_algo = it->second.first;
    h.mat_algo  = it->second.second;
    return h;
}

// ── Plan-kind set + skip-by-n/a ───────────────────────────────────

std::unordered_set<int> kind_set_of(const gamax::exec::ExecPlan& ep) {
    std::unordered_set<int> s;
    for (const auto& step : ep.steps) s.insert(static_cast<int>(step.kind));
    return s;
}
bool has_kind(const std::unordered_set<int>& s, gamax::exec::StepKind k) {
    return s.count(static_cast<int>(k)) > 0;
}

// "n/a" axis value matches a query that lacks the kind; non-"n/a"
// matches one that has it. Returns nullopt on match (not skipped),
// or a reason string when this cell should be skipped.
std::optional<std::string> mismatched(
    bool axis_set, bool kind_present, const char* name)
{
    if (axis_set && !kind_present) {
        return std::string{name} + "=set,plan_lacks_" + name;
    }
    if (!axis_set && kind_present) {
        return std::string{name} + "=n/a,plan_has_" + name;
    }
    return std::nullopt;
}

// CSV-safe a string by replacing commas/newlines with spaces.
std::string csv_safe(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == ',' || c == '\n' || c == '\r') out.push_back(' ');
        else out.push_back(c);
    }
    return out;
}

struct Args {
    int scale_factor {1};
    std::string out_path;
    int query_min {1};
    int query_max {22};
    bool quiet {false};
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            a.scale_factor = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            a.out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--query-min") == 0 && i + 1 < argc) {
            a.query_min = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--query-max") == 0 && i + 1 < argc) {
            a.query_max = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            a.quiet = true;
        } else {
            throw std::runtime_error(
                "tpch_ir_axis_probe: unknown arg: " + std::string(argv[i]));
        }
    }
    if (a.out_path.empty()) {
        throw std::runtime_error("--out <path> is required");
    }
    return a;
}

// Axis value lists. Match tpch_ir_nvbench.cu.
const std::vector<std::string> kFilterVals  = {"fuse", "nofuse", "n/a"};
const std::vector<std::string> kProjectVals = {"fuse", "nofuse", "n/a"};
const std::vector<std::string> kSortVals    = {"skf", "no_skf", "n/a"};
const std::vector<std::string> kGroupByVals = {
    "hash", "sort_gftr", "sort_gftr_opt", "sort_gfur",
    "partition_gftr", "partition_gfur", "n/a"};
const std::vector<std::string> kEqJoinVals  = {
    "hash_gfur", "hash_gftr",
    "sort_merge_gfur", "sort_merge_gftr",
    "partition_hash_gfur", "partition_hash_gftr",
    "n/a"};

} // namespace

int main(int argc, char** argv) try {
    auto args = parse_args(argc, argv);
    const bool reset_pool = std::getenv("GAMAX_PROBE_RESET_POOL") != nullptr;

    std::ofstream out(args.out_path);
    if (!out) {
        throw std::runtime_error("Failed to open output: " + args.out_path);
    }
    out << "query,scale,filter,project,sort,group_by,eq_join,status,error\n";

    auto& catalog = shared_catalog();
    auto& reader  = shared_reader(args.scale_factor);
    gamax::tpch::IrSourceResolver resolver(reader, catalog);
    gamax::exec::SourceResolver src_fn =
        [&resolver](const std::string& name,
                    const std::vector<std::string>& cols) {
            return resolver(name, cols);
        };

    int64_t pass_n = 0, fail_n = 0, skip_n = 0;
    auto t_start = std::chrono::steady_clock::now();

    for (int qid = args.query_min; qid <= args.query_max; ++qid) {
        // Build IR + apply rules once per query.
        gamax::ir::IdGenerator gen;
        gamax::ir::QueryBuilder qb(catalog, gen);
        auto plan = build_query(qid, qb);
        plan = gamax::ir::apply_rules(plan);

        // Lower with empty options once to extract the StepKind set
        // for skip-by-n/a. Reused across all axis combos for this query.
        gamax::exec::ExecPlan inspect_ep =
            gamax::ir::lower_to_exec_plan(plan, catalog);
        auto kinds = kind_set_of(inspect_ep);
        using K = gamax::exec::StepKind;
        const bool h_filter   = has_kind(kinds, K::Filter);
        const bool h_project  = has_kind(kinds, K::Project);
        const bool h_sort     = has_kind(kinds, K::Sort) || has_kind(kinds, K::TopK);
        const bool h_group_by = has_kind(kinds, K::GroupBy);
        const bool h_eq_join  = has_kind(kinds, K::EqJoin);

        for (const auto& fv : kFilterVals)
        for (const auto& pv : kProjectVals)
        for (const auto& sv : kSortVals)
        for (const auto& gv : kGroupByVals)
        for (const auto& jv : kEqJoinVals) {
            auto fh = parse_filter(fv);
            auto ph = parse_project(pv);
            auto sh = parse_sort(sv);
            auto gh = parse_group_by(gv);
            auto jh = parse_eq_join(jv);

            // Skip-by-n/a (silently — SKIP rows aren't emitted).
            if (mismatched(fh.has_value(), h_filter,   "filter"))   { ++skip_n; continue; }
            if (mismatched(ph.has_value(), h_project,  "project"))  { ++skip_n; continue; }
            if (mismatched(sh.has_value(), h_sort,     "sort"))     { ++skip_n; continue; }
            if (mismatched(gh.has_value(), h_group_by, "group_by")) { ++skip_n; continue; }
            if (mismatched(jh.has_value(), h_eq_join,  "eq_join"))  { ++skip_n; continue; }

            // Re-lower with overrides for this cell.
            gamax::ir::LowerOptions opts;
            opts.filter   = fh;
            opts.project  = ph;
            opts.sort     = sh;
            opts.top_k    = sh;       // share with Sort axis
            opts.group_by = gh;
            opts.eq_join  = jh;

            std::string status = "pass";
            std::string err;
            try {
                if (reset_pool) release_all_mem();
                auto ep = gamax::ir::lower_to_exec_plan(plan, catalog, opts);
                gamax::tpch::tpch_params params(args.scale_factor);
                gamax::phys::Context ctx{params.stream};
                auto result = gamax::exec::execute(ep, src_fn, ctx);
                cudaStreamSynchronize(params.stream);
                (void)result;
                ++pass_n;
            } catch (const std::exception& e) {
                status = "fail";
                err = csv_safe(e.what());
                ++fail_n;
            }

            out << qid << ',' << args.scale_factor << ',' << fv << ','
                << pv << ',' << sv << ',' << gv << ',' << jv << ','
                << status << ',' << err << '\n';

            if (!args.quiet && (pass_n + fail_n) % 100 == 0) {
                std::cerr << "[probe] " << (pass_n + fail_n)
                          << " runs (pass=" << pass_n
                          << " fail=" << fail_n
                          << " skip=" << skip_n << ")\n";
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                        t_end - t_start).count();
    std::cerr << "[probe] done: pass=" << pass_n
              << " fail=" << fail_n
              << " skip=" << skip_n
              << " (" << elapsed_s << " s, "
              << args.out_path << ")\n";
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "tpch_ir_axis_probe: " << e.what() << "\n";
    return 1;
}
