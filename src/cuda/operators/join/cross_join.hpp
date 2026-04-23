#pragma once

#include <memory>
#include <cuda_runtime_api.h>

#include "memory/table.hpp"

namespace gamax::join {

// Cartesian product of `left` and `right`. Output is
// `left.num_rows() * right.num_rows()` rows by
// `left.num_columns() + right.num_columns()` columns, in the schema
// order `[L_cols..., R_cols...]`. Row i of the output is the pair
// `(left[i / R.num_rows()], right[i % R.num_rows()])`, so the left
// side's row index advances slowest.
//
// Implementation: materialise two tuple-id columns (`i / n`, `i % n`
// for `n = right.num_rows()`) and gather each side's columns through
// them. An all-iterator variant that fuses the division/modulo into
// each column's gather is possible but offers marginal savings versus
// materialising two `size_type` columns of length `m*n` — the output
// table itself already occupies O(m*n) per data column — so we opt
// for the simpler shape that reuses the existing
// `join::detail::simple_materialize` path (STRING-safe via
// `gather_strings`).
std::unique_ptr<gamax::table> cross_join(
    gamax::table_view left,
    gamax::table_view right,
    cudaStream_t stream = cudaStreamDefault);

} // namespace gamax::join
