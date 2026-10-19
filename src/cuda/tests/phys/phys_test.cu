// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "expr/expr_fast.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/select/predicate.hpp"
#include "phys/phys.hpp"
#include "string/string_column.hpp"

namespace phys = gamax::phys;
using gamax::data_type;
using gamax::type_id;
using gamax::expression::ExprTree;

namespace {
data_type t(type_id id) { return data_type{id}; }

template <typename T>
gamax::column_ptr mk_col(const std::vector<T>& v) {
    return gamax::make_column_from_host(
        const_cast<T*>(v.data()),
        static_cast<gamax::size_type>(v.size()),
        cudaStreamDefault);
}

gamax::column_ptr mk_string_col(const std::vector<std::string>& ss) {
    std::vector<char> flat;
    std::vector<int32_t> lens;
    for (const auto& s : ss) {
        lens.push_back(static_cast<int32_t>(s.size()));
        flat.insert(flat.end(), s.begin(), s.end());
    }
    auto ptr = gamax::make_string_column(
        flat.data(), lens.data(),
        static_cast<gamax::size_type>(ss.size()),
        cudaStreamDefault, gamax::ARROW_STRING);
    return std::unique_ptr<gamax::column>(std::move(ptr));
}

phys::Context default_ctx() { return phys::Context{cudaStreamDefault}; }
} // namespace

// ── reorder: the B7 proof ─────────────────────────────────────────

TEST(PhysReorder, NumericSubsetAndReorder) {
    std::vector<int32_t> a = {1, 2, 3, 4};
    std::vector<double>  b = {1.1, 2.2, 3.3, 4.4};
    std::vector<int64_t> c = {10, 20, 30, 40};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    cols.push_back(mk_col(b));
    cols.push_back(mk_col(c));
    auto tbl = gamax::make_table(std::move(cols));

    auto out = phys::reorder(tbl->view(), {2, 0}, default_ctx());
    ASSERT_EQ(out->num_columns(), 2);
    ASSERT_EQ(out->num_rows(), 4);
    auto col0 = gamax::copy_column_to_host<int64_t>(out->view()[0], cudaStreamDefault);
    auto col1 = gamax::copy_column_to_host<int32_t>(out->view()[1], cudaStreamDefault);
    EXPECT_EQ(col0[0], 10); EXPECT_EQ(col0[3], 40);
    EXPECT_EQ(col1[0], 1);  EXPECT_EQ(col1[3], 4);
}

TEST(PhysReorder, PassesSTRINGColumnsThrough) {
    // B7: the previous emitter's reshape paths used project::project
    // which refuses STRING. phys::reorder must round-trip STRING.
    std::vector<int32_t> a = {1, 2, 3};
    std::vector<std::string> s = {"x", "yy", "zzz"};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    cols.push_back(mk_string_col(s));
    auto tbl = gamax::make_table(std::move(cols));

    auto out = phys::reorder(tbl->view(), {1, 0}, default_ctx());
    ASSERT_EQ(out->num_columns(), 2);
    ASSERT_EQ(out->num_rows(), 3);
    EXPECT_EQ(out->view()[0].type().id(), type_id::STRING);
    EXPECT_EQ(out->view()[1].type().id(), type_id::INT32);
    auto ints = gamax::copy_column_to_host<int32_t>(out->view()[1], cudaStreamDefault);
    EXPECT_EQ(ints[0], 1);
    EXPECT_EQ(ints[2], 3);
}

TEST(PhysReorder, RepeatedOrdinalProducesIndependentCopies) {
    std::vector<int32_t> a = {7, 8, 9};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));

    auto out = phys::reorder(tbl->view(), {0, 0, 0}, default_ctx());
    ASSERT_EQ(out->num_columns(), 3);
    auto c0 = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    auto c2 = gamax::copy_column_to_host<int32_t>(out->view()[2], cudaStreamDefault);
    EXPECT_EQ(c0[0], 7);
    EXPECT_EQ(c2[2], 9);
}

TEST(PhysReorder, RejectsOutOfRangeOrdinal) {
    std::vector<int32_t> a = {1};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));
    EXPECT_THROW(phys::reorder(tbl->view(), {5}, default_ctx()),
                 std::runtime_error);
}

// ── filter: heterogeneous compact ─────────────────────────────────

TEST(PhysFilter, NumericOnly) {
    std::vector<int32_t> a = {1, 5, 10, 15};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));

    auto pred = gamax::select::Predicate::expr(
        ExprTree::greater_than(
            ExprTree::column(0, t(type_id::INT32)),
            ExprTree::constant(gamax::scalar(int32_t{7}))));
    auto out = phys::filter(tbl->view(), pred, default_ctx());
    ASSERT_EQ(out->num_rows(), 2);
    auto kept = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    EXPECT_EQ(kept[0], 10);
    EXPECT_EQ(kept[1], 15);
}

TEST(PhysFilter, WithStringColumnPayload) {
    // Filter on numeric column; STRING payload must survive.
    std::vector<int32_t> a = {1, 2, 3, 4};
    std::vector<std::string> s = {"w", "xx", "yyy", "zzzz"};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    cols.push_back(mk_string_col(s));
    auto tbl = gamax::make_table(std::move(cols));

    auto pred = gamax::select::Predicate::expr(
        ExprTree::greater_equal(
            ExprTree::column(0, t(type_id::INT32)),
            ExprTree::constant(gamax::scalar(int32_t{3}))));
    auto out = phys::filter(tbl->view(), pred, default_ctx());
    ASSERT_EQ(out->num_rows(), 2);
    ASSERT_EQ(out->num_columns(), 2);
    EXPECT_EQ(out->view()[1].type().id(), type_id::STRING);
}

// Selective-output overload: the predicate references ordinal 0 but
// the consumer only wants column 1 materialized.
TEST(PhysFilter, SelectiveOutputDropsPredicateOnlyColumn) {
    std::vector<int32_t> a = {1, 2, 3, 4, 5};  // predicate column
    std::vector<double>  b = {1.5, 2.5, 3.5, 4.5, 5.5};  // payload
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    cols.push_back(mk_col(b));
    auto tbl = gamax::make_table(std::move(cols));

    auto pred = gamax::select::Predicate::expr(
        ExprTree::greater_than(
            ExprTree::column(0, t(type_id::INT32)),
            ExprTree::constant(gamax::scalar(int32_t{2}))));
    auto out = phys::filter(
        tbl->view(), pred, std::vector<gamax::size_type>{1}, default_ctx());
    ASSERT_EQ(out->num_columns(), 1);
    ASSERT_EQ(out->num_rows(), 3);
    EXPECT_EQ(out->view()[0].type().id(), type_id::FLOAT64);
    auto kept = gamax::copy_column_to_host<double>(out->view()[0], cudaStreamDefault);
    EXPECT_DOUBLE_EQ(kept[0], 3.5);
    EXPECT_DOUBLE_EQ(kept[1], 4.5);
    EXPECT_DOUBLE_EQ(kept[2], 5.5);
}

// Selective-output also drops a STRING column referenced only by the
// predicate. Closes the most interesting case: a predicate-only STRING
// column is expensive to gather, and pruning it should avoid the
// gather entirely.
TEST(PhysFilter, SelectiveOutputDropsStringPredicateColumn) {
    std::vector<int32_t> a = {10, 20, 30, 40};  // payload
    std::vector<std::string> s =                // predicate column
        {"apple", "banana", "cherry", "date"};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    cols.push_back(mk_string_col(s));
    auto tbl = gamax::make_table(std::move(cols));

    // Predicate: any numeric check; we keep the shape simple to
    // verify materialization selection, not predicate expressivity.
    auto pred = gamax::select::Predicate::expr(
        ExprTree::less_than(
            ExprTree::column(0, t(type_id::INT32)),
            ExprTree::constant(gamax::scalar(int32_t{35}))));
    auto out = phys::filter(
        tbl->view(), pred, std::vector<gamax::size_type>{0}, default_ctx());
    // Only column 0 materialized — no STRING column in the result.
    ASSERT_EQ(out->num_columns(), 1);
    ASSERT_EQ(out->num_rows(), 3);
    EXPECT_EQ(out->view()[0].type().id(), type_id::INT32);
    auto kept = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    EXPECT_EQ(kept[0], 10); EXPECT_EQ(kept[1], 20); EXPECT_EQ(kept[2], 30);
}

// Empty `output_ordinals` is equivalent to the 3-arg overload.
TEST(PhysFilter, SelectiveOutputEmptyMeansAll) {
    std::vector<int32_t> a = {1, 5, 10, 15};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));

    auto pred = gamax::select::Predicate::expr(
        ExprTree::greater_than(
            ExprTree::column(0, t(type_id::INT32)),
            ExprTree::constant(gamax::scalar(int32_t{7}))));
    auto out = phys::filter(
        tbl->view(), pred, std::vector<gamax::size_type>{}, default_ctx());
    ASSERT_EQ(out->num_columns(), 1);
    ASSERT_EQ(out->num_rows(), 2);
}

// 0-row input with selective output → 0-row output of the requested
// shape. Confirms the empty-input short-circuit respects ordinals.
TEST(PhysFilter, SelectiveOutputEmptyInput) {
    std::vector<int32_t> a = {};
    std::vector<double>  b = {};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    cols.push_back(mk_col(b));
    auto tbl = gamax::make_table(std::move(cols));

    auto pred = gamax::select::Predicate::expr(
        ExprTree::greater_than(
            ExprTree::column(0, t(type_id::INT32)),
            ExprTree::constant(gamax::scalar(int32_t{0}))));
    auto out = phys::filter(
        tbl->view(), pred, std::vector<gamax::size_type>{1}, default_ctx());
    ASSERT_EQ(out->num_columns(), 1);
    ASSERT_EQ(out->num_rows(), 0);
    EXPECT_EQ(out->view()[0].type().id(), type_id::FLOAT64);
}

// Out-of-range ordinal → clear error.
TEST(PhysFilter, SelectiveOutputRejectsOutOfRangeOrdinal) {
    std::vector<int32_t> a = {1, 2, 3};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));

    auto pred = gamax::select::Predicate::expr(
        ExprTree::greater_than(
            ExprTree::column(0, t(type_id::INT32)),
            ExprTree::constant(gamax::scalar(int32_t{0}))));
    EXPECT_THROW(
        phys::filter(
            tbl->view(), pred,
            std::vector<gamax::size_type>{1}, default_ctx()),
        std::runtime_error);
}

// ── project: thin wrapper around project::project ─────────────────

TEST(PhysProject, ComputesExpression) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {10.0, 20.0, 30.0};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    cols.push_back(mk_col(b));
    auto tbl = gamax::make_table(std::move(cols));

    std::vector<gamax::phys::ProjectItem> items;
    items.emplace_back(ExprTree::multiply(
        ExprTree::column(0, t(type_id::FLOAT64)),
        ExprTree::column(1, t(type_id::FLOAT64))));
    auto out = phys::project(tbl->view(), items, default_ctx());
    ASSERT_EQ(out->num_columns(), 1);
    auto result = gamax::copy_column_to_host<double>(out->view()[0], cudaStreamDefault);
    EXPECT_DOUBLE_EQ(result[0], 10.0);
    EXPECT_DOUBLE_EQ(result[2], 90.0);
}

// ── eq_join / semi_join ───────────────────────────────────────────

TEST(PhysEqJoin, SingleKeyMatches) {
    std::vector<int32_t> lk = {1, 2, 3};
    std::vector<double>  lv = {1.5, 2.5, 3.5};
    std::vector<int32_t> rk = {2, 3, 4};
    std::vector<int32_t> rv = {20, 30, 40};

    std::vector<gamax::column_ptr> lc;
    lc.push_back(mk_col(lk)); lc.push_back(mk_col(lv));
    std::vector<gamax::column_ptr> rc;
    rc.push_back(mk_col(rk)); rc.push_back(mk_col(rv));
    auto L = gamax::make_table(std::move(lc));
    auto R = gamax::make_table(std::move(rc));

    auto out = phys::eq_join(
        L->view(), {0}, R->view(), {0}, default_ctx());
    // [shared_key, L_rest..., R_rest...] — 3 columns.
    ASSERT_EQ(out->num_columns(), 3);
    EXPECT_EQ(out->num_rows(), 2);  // matching keys: 2, 3
}

TEST(PhysEqJoin, RejectsMultiKeyUntilLandingFusion) {
    std::vector<int32_t> a = {1, 2};
    std::vector<int32_t> b = {3, 4};
    std::vector<gamax::column_ptr> lc;
    lc.push_back(mk_col(a)); lc.push_back(mk_col(b));
    auto L = gamax::make_table(std::move(lc));
    std::vector<gamax::column_ptr> rc;
    rc.push_back(mk_col(a)); rc.push_back(mk_col(b));
    auto R = gamax::make_table(std::move(rc));
    EXPECT_THROW(
        phys::eq_join(L->view(), {0, 1}, R->view(), {0, 1}, default_ctx()),
        std::runtime_error);
}

TEST(PhysEqJoin, StringPayloadColumnsPassThrough) {
    // Repros the B7-in-join case that the old reshape_inner_join_back
    // silently failed at (project::project over STRING columns).
    std::vector<int32_t> lk = {1, 2};
    std::vector<std::string> ln = {"alpha", "beta"};
    std::vector<gamax::column_ptr> lc;
    lc.push_back(mk_col(lk)); lc.push_back(mk_string_col(ln));
    auto L = gamax::make_table(std::move(lc));

    std::vector<int32_t> rk = {1, 2};
    std::vector<int32_t> rv = {10, 20};
    std::vector<gamax::column_ptr> rc;
    rc.push_back(mk_col(rk)); rc.push_back(mk_col(rv));
    auto R = gamax::make_table(std::move(rc));

    auto out = phys::eq_join(
        L->view(), {0}, R->view(), {0}, default_ctx());
    ASSERT_EQ(out->num_columns(), 3);
    EXPECT_EQ(out->num_rows(), 2);
    // Column order: [shared_key, L_name (STRING), R_value (INT32)].
    EXPECT_EQ(out->view()[1].type().id(), type_id::STRING);
}

TEST(PhysSemiJoin, KeepsLeftOnly) {
    std::vector<int32_t> lk = {1, 2, 3, 4};
    std::vector<double>  lv = {1.0, 2.0, 3.0, 4.0};
    std::vector<int32_t> rk = {2, 4};
    std::vector<int32_t> rv = {99, 99};
    std::vector<gamax::column_ptr> lc;
    lc.push_back(mk_col(lk)); lc.push_back(mk_col(lv));
    std::vector<gamax::column_ptr> rc;
    rc.push_back(mk_col(rk)); rc.push_back(mk_col(rv));
    auto L = gamax::make_table(std::move(lc));
    auto R = gamax::make_table(std::move(rc));

    auto out = phys::semi_join(
        L->view(), {0}, R->view(), {0}, default_ctx());
    // Semi preserves left schema (reshaped): 2 cols = [key, lv].
    ASSERT_EQ(out->num_columns(), 2);
    EXPECT_EQ(out->num_rows(), 2);
}

// ── Outer join: still declared-but-not-implemented ────────────────
// Anti-join is now wired end-to-end (see tests/join/anti_join_test.cu
// and phys::anti_join); only left_outer_join remains unimplemented.

TEST(PhysJoins, OuterJoinStillThrows) {
    std::vector<int32_t> a = {1};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    cols.push_back(mk_col(a));
    auto L = gamax::make_table(std::move(cols));
    std::vector<gamax::column_ptr> cols2;
    cols2.push_back(mk_col(a));
    cols2.push_back(mk_col(a));
    auto R = gamax::make_table(std::move(cols2));

    EXPECT_THROW(phys::left_outer_join(L->view(), {0}, R->view(), {0}, default_ctx()),
                 std::runtime_error);
}

// ── CrossJoin (Phase 1.6, fake-key trick) ─────────────────────────

TEST(PhysCrossJoin, CartesianProduct) {
    // L has 2 rows (x=1,2), R has 3 rows (y=10,20,30).
    // Expected: 6 rows, schema [x, y].
    std::vector<int32_t> lx = {1, 2};
    std::vector<int32_t> ry = {10, 20, 30};
    std::vector<gamax::column_ptr> lc; lc.push_back(mk_col(lx));
    std::vector<gamax::column_ptr> rc; rc.push_back(mk_col(ry));
    auto L = gamax::make_table(std::move(lc));
    auto R = gamax::make_table(std::move(rc));

    auto out = phys::cross_join(L->view(), R->view(), default_ctx());
    EXPECT_EQ(out->num_columns(), 2);
    EXPECT_EQ(out->num_rows(), 6);
}

TEST(PhysCrossJoin, WithSTRINGPayload) {
    std::vector<int32_t> lx = {1, 2};
    std::vector<std::string> rn = {"a", "b", "c"};
    std::vector<gamax::column_ptr> lc; lc.push_back(mk_col(lx));
    std::vector<gamax::column_ptr> rc; rc.push_back(mk_string_col(rn));
    auto L = gamax::make_table(std::move(lc));
    auto R = gamax::make_table(std::move(rc));

    auto out = phys::cross_join(L->view(), R->view(), default_ctx());
    EXPECT_EQ(out->num_columns(), 2);
    EXPECT_EQ(out->num_rows(), 6);
    EXPECT_EQ(out->view()[1].type().id(), type_id::STRING);
}

TEST(PhysCrossJoin, EmptyEitherSide) {
    std::vector<int32_t> full = {1, 2, 3};
    std::vector<gamax::column_ptr> fc; fc.push_back(mk_col(full));
    auto F = gamax::make_table(std::move(fc));
    auto E = gamax::make_table({});  // 0 columns is different; use 0-row

    std::vector<int32_t> empty_data;
    std::vector<gamax::column_ptr> ec; ec.push_back(mk_col(empty_data));
    auto Empty = gamax::make_table(std::move(ec));

    auto out1 = phys::cross_join(F->view(), Empty->view(), default_ctx());
    EXPECT_EQ(out1->num_rows(), 0);
    EXPECT_EQ(out1->num_columns(), 2);
    auto out2 = phys::cross_join(Empty->view(), F->view(), default_ctx());
    EXPECT_EQ(out2->num_rows(), 0);
    EXPECT_EQ(out2->num_columns(), 2);
}

// ── Empty-input short-circuits (B8) ───────────────────────────────

namespace {
std::unique_ptr<gamax::table> make_empty_table_of_types(
    const std::vector<gamax::data_type>& types)
{
    std::vector<gamax::column_ptr> cols;
    for (auto t : types) {
        if (t.id() == type_id::STRING) {
            cols.push_back(mk_string_col({}));
        } else {
            // Numeric 0-row column via make_column.
            cols.push_back(gamax::make_column(
                t, 0, cudaStreamDefault,
                gamax::buffer_init::uninitialized));
        }
    }
    return gamax::make_table(std::move(cols));
}
} // namespace

TEST(PhysEmptyInput, FilterZeroRowIn) {
    auto tbl = make_empty_table_of_types({t(type_id::INT32), t(type_id::FLOAT64)});
    auto pred = gamax::select::Predicate::expr(
        ExprTree::greater_than(
            ExprTree::column(0, t(type_id::INT32)),
            ExprTree::constant(gamax::scalar(int32_t{0}))));
    auto out = phys::filter(tbl->view(), pred, default_ctx());
    EXPECT_EQ(out->num_rows(), 0);
    EXPECT_EQ(out->num_columns(), 2);
}

TEST(PhysEmptyInput, ScalarAggregateProducesOneRowOfZeros) {
    auto tbl = make_empty_table_of_types({t(type_id::INT32)});
    auto out = phys::scalar_aggregate(
        tbl->view(),
        {{0, gamax::groupby::aggregation_id::SUM}},
        default_ctx());
    ASSERT_EQ(out->num_rows(), 1);
    ASSERT_EQ(out->num_columns(), 1);
    auto v = gamax::copy_column_to_host<int64_t>(out->view()[0], cudaStreamDefault);
    EXPECT_EQ(v[0], 0);
}

TEST(PhysEmptyInput, GroupByProducesZeroRows) {
    auto keys = make_empty_table_of_types({t(type_id::INT32)});
    auto vals = make_empty_table_of_types({t(type_id::FLOAT64)});
    auto out = phys::group_by(
        keys->view(), vals->view(),
        {{0, gamax::groupby::aggregation_id::SUM}},
        default_ctx());
    EXPECT_EQ(out->num_rows(), 0);
    EXPECT_EQ(out->num_columns(), 2);  // key + sum output
}

TEST(PhysEmptyInput, EqJoinEitherSideEmpty) {
    auto full = make_empty_table_of_types({t(type_id::INT32), t(type_id::INT32)});
    std::vector<int32_t> a = {1, 2};
    std::vector<int32_t> b = {3, 4};
    std::vector<gamax::column_ptr> rc;
    rc.push_back(mk_col(a));
    rc.push_back(mk_col(b));
    auto R = gamax::make_table(std::move(rc));

    auto out = phys::eq_join(full->view(), {0}, R->view(), {0}, default_ctx());
    EXPECT_EQ(out->num_rows(), 0);
    // [shared_key, L_rest, R_rest] = 3 columns.
    EXPECT_EQ(out->num_columns(), 3);
}

TEST(PhysEmptyInput, SortAndTopK) {
    auto tbl = make_empty_table_of_types({t(type_id::INT32)});
    auto s = phys::sort(tbl->view(),
        {{0, gamax::sort_order::ASCENDING}}, default_ctx());
    EXPECT_EQ(s->num_rows(), 0);
    auto k = phys::top_k(tbl->view(),
        {{0, gamax::sort_order::DESCENDING}}, 5, default_ctx());
    EXPECT_EQ(k->num_rows(), 0);
}

TEST(PhysEmptyInput, ReorderZeroRows) {
    auto tbl = make_empty_table_of_types(
        {t(type_id::INT32), t(type_id::STRING), t(type_id::FLOAT64)});
    auto out = phys::reorder(tbl->view(), {2, 0, 1}, default_ctx());
    EXPECT_EQ(out->num_rows(), 0);
    EXPECT_EQ(out->num_columns(), 3);
    // ordinals {2, 0, 1}: out[0]=input[2]=FLOAT64, out[1]=input[0]=INT32,
    // out[2]=input[1]=STRING.
    EXPECT_EQ(out->view()[0].type().id(), type_id::FLOAT64);
    EXPECT_EQ(out->view()[1].type().id(), type_id::INT32);
    EXPECT_EQ(out->view()[2].type().id(), type_id::STRING);
}

// ── aggregate / group_by ──────────────────────────────────────────

TEST(PhysAggregate, ScalarSum) {
    std::vector<int32_t> v = {10, 20, 30};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(v));
    auto tbl = gamax::make_table(std::move(cols));

    auto out = phys::scalar_aggregate(
        tbl->view(),
        {{0, gamax::groupby::aggregation_id::SUM}},
        default_ctx());
    ASSERT_EQ(out->num_rows(), 1);
    auto sum = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    EXPECT_EQ(sum[0], 60);
}

TEST(PhysAggregate, GroupBySum) {
    std::vector<int32_t> k = {1, 1, 2, 2, 3};
    std::vector<int32_t> v = {10, 20, 30, 40, 50};
    std::vector<gamax::column_ptr> key_cols; key_cols.push_back(mk_col(k));
    std::vector<gamax::column_ptr> val_cols; val_cols.push_back(mk_col(v));
    auto keys = gamax::make_table(std::move(key_cols));
    auto vals = gamax::make_table(std::move(val_cols));

    auto out = phys::group_by(
        keys->view(), vals->view(),
        {{0, gamax::groupby::aggregation_id::SUM}},
        default_ctx());
    // 3 distinct keys → 3 groups → 3 rows.
    EXPECT_EQ(out->num_rows(), 3);
    EXPECT_EQ(out->num_columns(), 2);  // key + sum
}

// ── sort / top_k ──────────────────────────────────────────────────

TEST(PhysSort, Ascending) {
    std::vector<int32_t> v = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<gamax::column_ptr> cols; cols.push_back(mk_col(v));
    auto tbl = gamax::make_table(std::move(cols));

    auto out = phys::sort(
        tbl->view(),
        {{0, gamax::sort_order::ASCENDING}},
        default_ctx());
    auto s = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    ASSERT_EQ(out->num_rows(), 8);
    EXPECT_EQ(s[0], 1);
    EXPECT_EQ(s[7], 9);
}

TEST(PhysTopK, DescendingTop3) {
    std::vector<int32_t> v = {3, 1, 4, 1, 5, 9, 2, 6};
    std::vector<gamax::column_ptr> cols; cols.push_back(mk_col(v));
    auto tbl = gamax::make_table(std::move(cols));

    auto out = phys::top_k(
        tbl->view(),
        {{0, gamax::sort_order::DESCENDING}},
        3,
        default_ctx());
    ASSERT_EQ(out->num_rows(), 3);
    auto s = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    EXPECT_EQ(s[0], 9);
    EXPECT_EQ(s[1], 6);
    EXPECT_EQ(s[2], 5);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
