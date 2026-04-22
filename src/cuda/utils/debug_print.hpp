#pragma once

#include <iosfwd>
#include <string>
#include <unordered_set>

#include "memory/table.hpp"

// Forward-declarations so this header stays lightweight. Callers that
// actually invoke `print_plans_if_debug` will include the IR / ExecPlan
// headers themselves.
namespace gamax::ir { class PlanNode; }
namespace gamax::exec { struct ExecPlan; }

// Lightweight debug helpers. Two kinds of users today:
//   1. The legacy TPC-H runners in `tpch/` — `PRINT_TABLE_PTR` in
//      `tpch/tpch.hpp` delegates here (stdout, unchanged behaviour).
//   2. The ExecPlan executor — when built with `-DGAMAX_PHYS_DEBUG` and
//      run with `GAMAX_PHYS_DEBUG=<kinds>`, each Step's output is
//      printed to stderr right after it's produced. See
//      `exec/executor.cpp` for the wiring.

namespace gamax::util::debug {

// Print a table's schema + the first `nrows` rows starting at `offset`.
// Passing `nrows == -1` prints every row from `offset` onward. Handles
// STRING columns via a dedicated branch (ordinary numeric columns go
// through `type_dispatcher`).
void print_table(
    const gamax::table_view& view,
    gamax::size_type nrows,
    gamax::size_type offset,
    std::ostream& os);

// Overload defaulting to std::cout — matches the pre-existing behaviour
// of `gamax::tpch::print_table` so callers don't have to pass the
// stream explicitly.
void print_table(
    const gamax::table_view& view,
    gamax::size_type nrows = -1,
    gamax::size_type offset = 0);

// Per-execute()-call filter for the phys-debug feature. Reads the
// `GAMAX_PHYS_DEBUG` and `GAMAX_PHYS_DEBUG_ROWS` environment variables
// once at construction and exposes accessors the executor queries per
// step. When `GAMAX_PHYS_DEBUG` is unset (or empty), the filter admits
// nothing — prints are skipped even when built with the compile-time
// macro defined.
//
// Values accepted by `GAMAX_PHYS_DEBUG`:
//   * empty / unset  → no prints.
//   * `all` or `*`   → every StepKind.
//   * comma-separated StepKind names (lower-case, matches the `kind_name`
//     strings returned by `exec/print.cpp`), e.g. `eq_join,group_by`.
// `GAMAX_PHYS_DEBUG_ROWS` overrides the per-print row count. Defaults
// to 10.
class PhysDebugFilter {
public:
    PhysDebugFilter();

    // True if the given kind name is in the admitted set. Comparison is
    // case-sensitive against the lower-case `StepKind` names used by
    // `kind_name` in `exec/print.cpp`.
    bool should_print(const std::string& kind_name) const;

    // True iff the filter was constructed in a mode where at least one
    // kind might be printed. Cheap short-circuit the executor can use
    // to skip both the env lookup and the per-kind check in the common
    // case.
    bool any_enabled() const { return any_enabled_; }

    gamax::size_type max_rows() const { return max_rows_; }

private:
    bool any_enabled_ {false};
    bool match_all_   {false};
    std::unordered_set<std::string> kinds_;
    gamax::size_type max_rows_ {10};
};

// One-shot plan dump. Prints the pre-rules IR, post-rules IR, and
// ExecPlan to `os` (default stderr) when *either* condition holds:
//   * `GAMAX_PHYS_DEBUG` is set and non-empty (any value), OR
//   * `GAMAX_DEBUG_PLAN` is set to exactly "1".
// Callers typically invoke this once, right after lowering to
// ExecPlan, so both IR snapshots are still live.
void print_plans_if_debug(
    const gamax::ir::PlanNode& pre_rules,
    const gamax::ir::PlanNode& post_rules,
    const gamax::exec::ExecPlan& ep,
    std::ostream& os);
void print_plans_if_debug(
    const gamax::ir::PlanNode& pre_rules,
    const gamax::ir::PlanNode& post_rules,
    const gamax::exec::ExecPlan& ep);

} // namespace gamax::util::debug
