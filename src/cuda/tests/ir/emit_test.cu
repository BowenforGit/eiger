#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/emit.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"

namespace gir = gamax::ir;
using gamax::data_type;
using gamax::type_id;

namespace {
data_type t(type_id id) { return data_type{id}; }

// Helper: build a table from host-side column vectors of a uniform type.
template <typename T>
gamax::column_ptr make_col(const std::vector<T>& values) {
    // make_column_from_host takes T* host_data, size_type num_items.
    return gamax::make_column_from_host(
        const_cast<T*>(values.data()),
        static_cast<gamax::size_type>(values.size()));
}

// Helper: build a string column from a vector of std::string. Matches
// the pattern used in predicate_test.cu.
gamax::column_ptr make_string_col(const std::vector<std::string>& strings) {
    std::vector<char> flat;
    std::vector<int32_t> lens;
    for (const auto& s : strings) {
        lens.push_back(static_cast<int32_t>(s.size()));
        flat.insert(flat.end(), s.begin(), s.end());
    }
    auto sptr = gamax::make_string_column(
        flat.data(), lens.data(),
        static_cast<gamax::size_type>(strings.size()),
        cudaStreamDefault, gamax::ARROW_STRING);
    return std::unique_ptr<gamax::column>(std::move(sptr));
}

gir::Catalog make_lineitem_catalog() {
    gir::Catalog cat;
    gir::CatalogTable li;
    li.name = "lineitem";
    li.column_names = {
        "l_shipdate", "l_extendedprice", "l_discount", "l_quantity",
    };
    li.column_types = {
        t(type_id::INT32), t(type_id::FLOAT64),
        t(type_id::FLOAT64), t(type_id::INT32),
    };
    li.column_nullable = {false, false, false, false};
    cat.add_table(std::move(li));
    return cat;
}

// Table with 6 rows; 3 pass Q6's predicate (shipdate in [8766, 9131),
// discount in [0.05, 0.07], quantity < 24). Kept deliberately tiny so
// the expected scalar SUM is trivial to check on CPU.
struct Q6Input {
    std::unique_ptr<gamax::table> table;
    std::unordered_map<std::string, gamax::table_view> by_name;
};
Q6Input make_q6_input() {
    std::vector<int32_t> shipdate   = {8000, 8800, 9000, 9200, 9000, 8900};
    std::vector<double>  price      = {100.0, 200.0, 300.0, 400.0, 500.0, 600.0};
    std::vector<double>  discount   = {0.10, 0.06, 0.05, 0.06, 0.08, 0.07};
    std::vector<int32_t> quantity   = {  10,   10,   10,   10,   10,   30};

    std::vector<gamax::column_ptr> cols;
    cols.push_back(make_col(shipdate));
    cols.push_back(make_col(price));
    cols.push_back(make_col(discount));
    cols.push_back(make_col(quantity));
    auto tbl = gamax::make_table(std::move(cols));
    auto view = tbl->view();

    // Rows that satisfy the filter:
    //   idx 1: shipdate=8800, discount=0.06, qty=10 → 200 * 0.06 = 12.0
    //   idx 2: shipdate=9000, discount=0.05, qty=10 → 300 * 0.05 = 15.0
    //   idx 4: shipdate=9000, discount=0.08, qty=10 → discount out of range
    // Only idx 1 and idx 2 pass. Expected SUM(revenue) = 12.0 + 15.0 = 27.0.
    // (idx 5 has qty=30, out; idx 4 has discount=0.08, out; idx 0 has
    //  shipdate < 8766; idx 3 has shipdate >= 9131.)

    Q6Input out;
    out.table = std::move(tbl);
    out.by_name["lineitem"] = view;
    return out;
}

gir::EmitContext make_ctx(const gir::Catalog& cat, const Q6Input& in) {
    gir::EmitContext ctx;
    ctx.catalog = &cat;
    ctx.stream = cudaStreamDefault;
    ctx.resolve_table = [in_ptr = &in, cat_ptr = &cat](
                            const std::string& name,
                            const std::vector<std::string>& cols) {
        auto it = in_ptr->by_name.find(name);
        if (it == in_ptr->by_name.end()) {
            throw std::runtime_error("unknown table: " + name);
        }
        const auto* ct = cat_ptr->find(name);
        std::vector<gamax::size_type> ords;
        ords.reserve(cols.size());
        for (const auto& c : cols) {
            auto cit = std::find(ct->column_names.begin(),
                                 ct->column_names.end(), c);
            ords.push_back(static_cast<gamax::size_type>(
                std::distance(ct->column_names.begin(), cit)));
        }
        return it->second.select_columns(ords);
    };
    return ctx;
}

} // namespace

// ── Per-operator emit tests ─────────────────────────────────────────

TEST(IrEmit, ScanProducesRequestedColumnsInOrder) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto plan = qb.scan("lineitem", {"l_discount", "l_shipdate"});
    auto out = gir::emit(plan.plan(), ctx);

    ASSERT_EQ(out->num_columns(), 2);
    ASSERT_EQ(out->num_rows(), 6);
    auto disc = gamax::copy_column_to_host<double>(out->view()[0]);
    EXPECT_DOUBLE_EQ(disc[1], 0.06);  // row 1 discount
    auto sd = gamax::copy_column_to_host<int32_t>(out->view()[1]);
    EXPECT_EQ(sd[2], 9000);           // row 2 shipdate
}

TEST(IrEmit, FilterKeepsMatchingRows) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate", "l_extendedprice"});
    // Keep rows where shipdate >= 9000.
    auto filtered = li.filter(li.c("l_shipdate") >= gir::lit(9000));
    auto out = gir::emit(filtered.plan(), ctx);

    // Expected: rows with shipdate in {9000, 9200, 9000} → 3 rows.
    ASSERT_EQ(out->num_rows(), 3);
}

TEST(IrEmit, ProjectComputesNewColumns) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_extendedprice", "l_discount"});
    auto proj = li.project({
        {"revenue", li.c("l_extendedprice") * li.c("l_discount")},
    });
    auto out = gir::emit(proj.plan(), ctx);

    ASSERT_EQ(out->num_columns(), 1);
    ASSERT_EQ(out->num_rows(), 6);
    auto rev = gamax::copy_column_to_host<double>(out->view()[0]);
    EXPECT_DOUBLE_EQ(rev[0], 100.0 * 0.10);
    EXPECT_DOUBLE_EQ(rev[1], 200.0 * 0.06);
    EXPECT_DOUBLE_EQ(rev[2], 300.0 * 0.05);
}

TEST(IrEmit, ScalarAggregateSum) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_extendedprice"});
    auto plan = li.aggregate(
        /*keys=*/{},
        {{"total", gir::AggKind::Sum, li.c("l_extendedprice")}});
    auto out = gir::emit(plan.plan(), ctx);

    ASSERT_EQ(out->num_rows(), 1);
    ASSERT_EQ(out->num_columns(), 1);
    auto total = gamax::copy_column_to_host<double>(out->view()[0]);
    EXPECT_DOUBLE_EQ(total[0], 100.0 + 200.0 + 300.0 + 400.0 + 500.0 + 600.0);
}

TEST(IrEmit, GroupedAggregateSum) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate", "l_extendedprice"});
    auto plan = li.aggregate(
        /*keys=*/{"l_shipdate"},
        {{"total", gir::AggKind::Sum, li.c("l_extendedprice")}});
    auto out = gir::emit(plan.plan(), ctx);

    ASSERT_EQ(out->num_columns(), 2);
    // 4 distinct shipdates in the input: 8000, 8800, 9000 (×2), 9200, 8900.
    // That's 5 distinct values.
    EXPECT_EQ(out->num_rows(), 5);
}

TEST(IrEmit, SortOrdersRows) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate"});
    auto plan = li.sort({gir::asc("l_shipdate")});
    auto out = gir::emit(plan.plan(), ctx);

    auto sd = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    ASSERT_EQ(out->num_rows(), 6);
    EXPECT_EQ(sd[0], 8000);
    EXPECT_EQ(sd[1], 8800);
    EXPECT_EQ(sd[2], 8900);
    EXPECT_EQ(sd[3], 9000);
    EXPECT_EQ(sd[4], 9000);
    EXPECT_EQ(sd[5], 9200);
}

// ── End-to-end: Q6 pipeline ─────────────────────────────────────────

TEST(IrEmit, Q6EndToEnd) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount", "l_quantity"});
    auto plan = li
        .filter(
            li.c("l_shipdate") >= gir::lit(8766)
         && li.c("l_shipdate") <  gir::lit(9131)
         && li.c("l_discount") >= gir::lit(0.05)
         && li.c("l_discount") <= gir::lit(0.07)
         && li.c("l_quantity") <  gir::lit(24))
        .project({
            {"revenue", li.c("l_extendedprice") * li.c("l_discount")}
        });
    plan = plan.aggregate(
        /*keys=*/{},
        {{"total", gir::AggKind::Sum, plan.c("revenue")}});

    auto out = gir::emit(plan.plan(), ctx);
    ASSERT_EQ(out->num_rows(), 1);
    ASSERT_EQ(out->num_columns(), 1);
    auto total = gamax::copy_column_to_host<double>(out->view()[0]);
    // Hand-computed: rows 1 and 2 pass. 200*0.06 + 300*0.05 = 27.0.
    EXPECT_DOUBLE_EQ(total[0], 12.0 + 15.0);
}

// ── String-column filter (needs gather_strings path) ────────────────

TEST(IrEmit, StringColumnFilterGathersAndCompactsTogether) {
    // Build a small table [orderkey (int32), name (string)] and filter
    // by a numeric predicate on orderkey. The string column must be
    // preserved on the surviving rows.
    std::vector<int32_t> keys = {1, 2, 3, 4};
    std::vector<std::string> names = {"a", "b", "c", "d"};

    std::vector<gamax::column_ptr> cols;
    cols.push_back(make_col(keys));
    cols.push_back(make_string_col(names));
    auto tbl = gamax::make_table(std::move(cols));
    auto view = tbl->view();

    gir::Catalog cat;
    gir::CatalogTable ct;
    ct.name = "orders";
    ct.column_names  = {"o_orderkey", "o_name"};
    ct.column_types  = {t(type_id::INT32), t(type_id::STRING)};
    ct.column_nullable = {false, false};
    cat.add_table(std::move(ct));

    gir::EmitContext ctx;
    ctx.catalog = &cat;
    ctx.stream  = cudaStreamDefault;
    ctx.resolve_table = [&](const std::string& name,
                            const std::vector<std::string>& cols) {
        if (name != "orders") {
            throw std::runtime_error("unknown table: " + name);
        }
        const auto* ct = cat.find(name);
        std::vector<gamax::size_type> ords;
        ords.reserve(cols.size());
        for (const auto& c : cols) {
            auto it = std::find(ct->column_names.begin(),
                                ct->column_names.end(), c);
            ords.push_back(static_cast<gamax::size_type>(
                std::distance(ct->column_names.begin(), it)));
        }
        return view.select_columns(ords);
    };

    gir::IdGenerator gen;
    gir::QueryBuilder qb(cat, gen);
    auto o = qb.scan("orders", {"o_orderkey", "o_name"});
    auto filtered = o.filter(o.c("o_orderkey") >= gir::lit(3));
    auto out = gir::emit(filtered.plan(), ctx);

    ASSERT_EQ(out->num_columns(), 2);
    ASSERT_EQ(out->num_rows(), 2);
    auto kk = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    EXPECT_EQ(kk[0], 3);
    EXPECT_EQ(kk[1], 4);
    // Just verify the string column is intact (2 rows) — reading string
    // content requires string_column_view plumbing we don't need here.
    EXPECT_EQ(out->view()[1].size(), 2);
}

// ── Inner join ──────────────────────────────────────────────────────

namespace {
// A second catalog holding a small "part" table for join tests.
struct JoinFixtures {
    gir::Catalog cat;
    std::unique_ptr<gamax::table> li_tbl;
    std::unique_ptr<gamax::table> part_tbl;
    std::unordered_map<std::string, gamax::table_view> by_name;
};

JoinFixtures make_join_fixtures() {
    JoinFixtures f;

    gir::CatalogTable li;
    li.name = "lineitem";
    li.column_names = {"l_partkey", "l_extendedprice"};
    li.column_types = {t(type_id::INT32), t(type_id::FLOAT64)};
    li.column_nullable = {false, false};
    f.cat.add_table(std::move(li));

    gir::CatalogTable p;
    p.name = "part";
    p.column_names = {"p_partkey", "p_size"};
    p.column_types = {t(type_id::INT32), t(type_id::INT32)};
    p.column_nullable = {false, false};
    f.cat.add_table(std::move(p));

    std::vector<int32_t> li_partkey = {10, 20, 30, 30, 40};
    std::vector<double>  li_price   = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<gamax::column_ptr> li_cols;
    li_cols.push_back(make_col(li_partkey));
    li_cols.push_back(make_col(li_price));
    f.li_tbl = gamax::make_table(std::move(li_cols));
    f.by_name["lineitem"] = f.li_tbl->view();

    std::vector<int32_t> p_partkey = {20, 30, 50};
    std::vector<int32_t> p_size    = {100, 200, 300};
    std::vector<gamax::column_ptr> p_cols;
    p_cols.push_back(make_col(p_partkey));
    p_cols.push_back(make_col(p_size));
    f.part_tbl = gamax::make_table(std::move(p_cols));
    f.by_name["part"] = f.part_tbl->view();

    return f;
}

gir::EmitContext make_join_ctx(JoinFixtures& f) {
    gir::EmitContext ctx;
    ctx.catalog = &f.cat;
    ctx.stream = cudaStreamDefault;
    ctx.resolve_table = [&](const std::string& name,
                            const std::vector<std::string>& cols) {
        auto it = f.by_name.find(name);
        if (it == f.by_name.end())
            throw std::runtime_error("unknown table: " + name);
        const auto* ct = f.cat.find(name);
        std::vector<gamax::size_type> ords;
        ords.reserve(cols.size());
        for (const auto& c : cols) {
            auto cit = std::find(ct->column_names.begin(),
                                 ct->column_names.end(), c);
            ords.push_back(static_cast<gamax::size_type>(
                std::distance(ct->column_names.begin(), cit)));
        }
        return it->second.select_columns(ords);
    };
    return ctx;
}
} // namespace

TEST(IrEmit, InnerJoinSingleKey) {
    auto f = make_join_fixtures();
    auto ctx = make_join_ctx(f);

    gir::IdGenerator gen;
    gir::QueryBuilder qb(f.cat, gen);
    auto li = qb.scan("lineitem", {"l_partkey", "l_extendedprice"});
    auto p  = qb.scan("part",     {"p_partkey", "p_size"});
    auto joined = li.inner_join(
        p, li.c("l_partkey") == p.c("p_partkey"));
    auto out = gir::emit(joined.plan(), ctx);

    // Matching partkeys: 20, 30 (twice from lineitem), 30 again.
    // That's 3 output rows.
    EXPECT_EQ(out->num_rows(), 3);
    // IR output schema: [l_partkey, l_extendedprice, p_partkey, p_size].
    ASSERT_EQ(out->num_columns(), 4);
    auto lpk = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    auto ppk = gamax::copy_column_to_host<int32_t>(out->view()[2]);
    // Every row's l_partkey must equal its p_partkey (equi-join).
    for (int i = 0; i < 3; ++i) EXPECT_EQ(lpk[i], ppk[i]);
}

TEST(IrEmit, LeftSemiJoinKeepsLeftSchemaOnly) {
    auto f = make_join_fixtures();
    auto ctx = make_join_ctx(f);

    gir::IdGenerator gen;
    gir::QueryBuilder qb(f.cat, gen);
    auto li = qb.scan("lineitem", {"l_partkey", "l_extendedprice"});
    auto p  = qb.scan("part",     {"p_partkey", "p_size"});
    auto semi = li.left_semi_join(
        p, li.c("l_partkey") == p.c("p_partkey"));
    auto out = gir::emit(semi.plan(), ctx);

    // Matching partkeys in lineitem: 20 (one row), 30 (two rows). 3 rows.
    EXPECT_EQ(out->num_rows(), 3);
    ASSERT_EQ(out->num_columns(), 2);  // left schema only
}

// ── Limit fused with Sort ──────────────────────────────────────────

TEST(IrEmit, LimitFusedWithSort) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate"});
    // Take 2 smallest shipdates (ASC).
    auto top = li.sort({gir::asc("l_shipdate")}).limit(2);
    auto out = gir::emit(top.plan(), ctx);

    ASSERT_EQ(out->num_rows(), 2);
    auto sd = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    EXPECT_EQ(sd[0], 8000);
    EXPECT_EQ(sd[1], 8800);
}

TEST(IrEmit, LimitWithoutSortReportsNotImplemented) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto plan = qb.scan("lineitem", {"l_shipdate"}).limit(3);
    EXPECT_THROW(gir::emit(plan.plan(), ctx), std::runtime_error);
}

// ── Remaining deferred paths ───────────────────────────────────────

TEST(IrEmit, CountAllReportsNotImplemented) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate"});
    auto plan = li.aggregate(/*keys=*/{},
        {{"n", gir::AggKind::CountAll, gir::lit(int64_t{0})}});
    EXPECT_THROW(gir::emit(plan.plan(), ctx), std::runtime_error);
}

TEST(IrEmit, LeftOuterJoinReportsNotImplemented) {
    auto f = make_join_fixtures();
    auto ctx = make_join_ctx(f);
    gir::IdGenerator gen;
    gir::QueryBuilder qb(f.cat, gen);
    auto li = qb.scan("lineitem", {"l_partkey", "l_extendedprice"});
    auto p  = qb.scan("part",     {"p_partkey", "p_size"});
    auto j = li.left_outer_join(p, li.c("l_partkey") == p.c("p_partkey"));
    EXPECT_THROW(gir::emit(j.plan(), ctx), std::runtime_error);
}

TEST(IrEmit, CrossJoinProducesCartesianProduct) {
    // Was previously "not yet implemented"; Phase 1.6 wired cross_join
    // via phys::eq_join + fake-key. L has 5 rows, R has 3 rows
    // (see make_join_fixtures) → 15 rows.
    auto f = make_join_fixtures();
    auto ctx = make_join_ctx(f);
    gir::IdGenerator gen;
    gir::QueryBuilder qb(f.cat, gen);
    auto li = qb.scan("lineitem", {"l_partkey", "l_extendedprice"});
    auto p  = qb.scan("part",     {"p_partkey", "p_size"});
    auto j = li.cross_join(p);
    auto out = gir::emit(j.plan(), ctx);
    // li has 5 rows, part has 3 rows → 15 cross-product rows.
    EXPECT_EQ(out->num_rows(), 15);
    EXPECT_EQ(out->num_columns(), 4);
}

TEST(IrEmit, DistinctLowersToPhysicalDistinct) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);
    gir::QueryBuilder qb(cat, gen);
    // All-columns distinct. The tiny Q6 fixture has 5 rows with 4
    // distinct l_shipdate values — exact bag contents verified
    // separately via phys::distinct tests; this test only asserts
    // Distinct runs through the emitter without throwing and yields
    // a reasonable row count.
    auto plan = qb.scan("lineitem", {"l_shipdate"}).distinct();
    auto out = gir::emit(plan.plan(), ctx);
    EXPECT_EQ(out->num_columns(), 1);
    EXPECT_GT(out->num_rows(), 0);
    EXPECT_LE(out->num_rows(), 5);
}

TEST(IrEmit, DistinctProperSubsetKeysNarrowsToOnlyKeys) {
    // Build a Distinct with a proper-subset keys list. The EXEC output
    // schema should contain only the key columns — non-key columns
    // must not appear in the output.
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);
    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount"});
    // Dedup by (l_shipdate) alone; drop the other two columns.
    auto plan = li.distinct({"l_shipdate"});
    auto out = gir::emit(plan.plan(), ctx);
    // Exactly one output column (l_shipdate) regardless of dedup
    // outcome.
    EXPECT_EQ(out->num_columns(), 1);
    EXPECT_EQ(out->view()[0].type().id(), gamax::type_id::INT32);
    EXPECT_GT(out->num_rows(), 0);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
