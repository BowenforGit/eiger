#pragma once

// Column-wise comparison of a gamax result against a pre-baked golden
// Parquet file (written by `script/gen_tpch_golden.py`). Keeps DuckDB
// out of the C++ build — the verifier only needs Arrow/Parquet, which
// are already linked for source ingestion.
//
// Type compatibility is positional: col[i] of the result must match
// col[i] of the golden. Per-type rules:
//   * INT32 / INT64 / BOOL8                     : exact.
//   * FLOAT32 / FLOAT64                         : |a - b| <= max(abs_tol, rel_tol * |b|).
//   * STRING                                    : exact.
// DATE goes through gamax's DATE32→INT32 YYYYMMDD convention on the
// gamax side; the golden script casts DuckDB DATE the same way, so
// both sides are INT32 here and the exact rule applies.

#include <ostream>
#include <string>

#include "memory/table.hpp"

namespace gamax::tpch {

struct VerifyOptions {
    double abs_tol = 0.01;    // TPC-H spec monetary tolerance
    double rel_tol = 1e-6;    // cover small drift on large magnitudes
    std::size_t max_diffs = 5; // cap diagnostic rows to keep output short
};

struct VerifyResult {
    bool ok {false};
    std::string diagnostic;  // populated on mismatch (row count, type,
                             // value); empty on OK.
};

// Load `golden_path` (Parquet), compare to `got` column-by-column, and
// return a verdict. Does not throw on mismatch — caller decides what
// to do with a failing result.
VerifyResult verify_result(
    gamax::table_view got,
    const std::string& golden_path,
    const VerifyOptions& opts = {});

} // namespace gamax::tpch
