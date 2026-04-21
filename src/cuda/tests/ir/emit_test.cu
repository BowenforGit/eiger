#include <gtest/gtest.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/emit.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"

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
    ctx.resolve_table = [in_ptr = &in](const std::string& name) {
        auto it = in_ptr->by_name.find(name);
        if (it == in_ptr->by_name.end()) {
            throw std::runtime_error("unknown table: " + name);
        }
        return it->second;
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

// ── NotImplemented errors for deferred physical paths ───────────────

TEST(IrEmit, LimitReportsNotImplemented) {
    gir::IdGenerator gen;
    auto cat = make_lineitem_catalog();
    auto in = make_q6_input();
    auto ctx = make_ctx(cat, in);

    gir::QueryBuilder qb(cat, gen);
    auto plan = qb.scan("lineitem", {"l_shipdate"}).limit(3);
    EXPECT_THROW(gir::emit(plan.plan(), ctx), std::runtime_error);
}

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

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
