#pragma once

// TPC-H queries expressed against the GaMax IR.
//
// This file is an expressivity test for `gamax::ir`: does the IR as it
// stands today let us *represent* the full TPC-H Q1..Q22 workload
// (decomposed per the Maximus reference plans)? Emit-ability is a
// separate concern — a number of these plans deliberately use IR nodes
// whose physical path isn't wired yet (cross_join for Q11, multi-key
// joins, FunctionCall scalars for year()/extract(), COUNT(*) / COUNT
// DISTINCT). What matters here is that the plan *builds* — that
// ColumnIds, schemas, and types line up end-to-end, and that every
// SQL construct these queries need has a corresponding IR surface.
//
// Q1..Q11 are provided here; Q12..Q22 will follow. When one of these
// plans fails to construct, it points at a real IR gap to close before
// TPC-H coverage is complete.

#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/plan_node.hpp"

namespace gamax::ir::tpch {

// Populate a Catalog with the TPC-H schemas used by Q1..Q22. Types
// follow the conventions already in the codebase:
//   * keys      -> INT64
//   * dates     -> INT32 (days since epoch)
//   * monetary  -> FLOAT64
//   * strings   -> STRING
//   * enums     -> STRING (returnflag, linestatus, orderstatus, ...)
// Columns are declared not-nullable (TPC-H has no NULLs).
Catalog make_tpch_catalog();

// Build the logical plan for TPC-H query N. Each takes a QueryBuilder
// whose catalog must have been produced by make_tpch_catalog() (or an
// equivalent); the builder's IdGenerator is advanced as each plan is
// constructed, so passing a single builder across all queries is fine
// and yields globally unique ColumnIds.
PlanPtr build_q1(const QueryBuilder& qb);
PlanPtr build_q2(const QueryBuilder& qb);
PlanPtr build_q3(const QueryBuilder& qb);
PlanPtr build_q4(const QueryBuilder& qb);
PlanPtr build_q5(const QueryBuilder& qb);
PlanPtr build_q6(const QueryBuilder& qb);
PlanPtr build_q7(const QueryBuilder& qb);
PlanPtr build_q8(const QueryBuilder& qb);
PlanPtr build_q9(const QueryBuilder& qb);
PlanPtr build_q10(const QueryBuilder& qb);
PlanPtr build_q11(const QueryBuilder& qb);

} // namespace gamax::ir::tpch
