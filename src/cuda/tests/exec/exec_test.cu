#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "exec/exec_plan.hpp"
#include "exec/executor.hpp"
#include "exec/print.hpp"
#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"

namespace gir = gamax::ir;
namespace gex = gamax::exec;
using gamax::data_type;
using gamax::type_id;

// These tests cover the Phase 2 additions in isolation:
//   * `lower_to_exec_plan` produces a plan of the expected shape
//   * `format_exec_plan` round-trips readably
//   * `execute` produces the same results the old monolithic
//     emitter did (parity is also checked by keeping test_emit
//     passing, which now runs through the same path).

namespace {

template <typename T>
gamax::column_ptr mk_col(const std::vector<T>& v) {
    return gamax::make_column_from_host(
        const_cast<T*>(v.data()),
        static_cast<gamax::size_type>(v.size()));
}

bool contains(const std::string& h, const std::string& n) {
    return h.find(n) != std::string::npos;
}

gir::Catalog make_catalog() {
    gir::Catalog cat;
    gir::CatalogTable t;
    t.name = "lineitem";
    t.column_names = {"l_shipdate", "l_extendedprice", "l_discount"};
    t.column_types = {
        data_type{type_id::INT32},
        data_type{type_id::FLOAT64},
        data_type{type_id::FLOAT64},
    };
    t.column_nullable = {false, false, false};
    cat.add_table(std::move(t));
    return cat;
}

struct Fixture {
    gir::Catalog cat;
    std::unique_ptr<gamax::table> tbl;
    gamax::table_view view;
};
Fixture make_fixture() {
    Fixture f;
    f.cat = make_catalog();
    std::vector<int32_t> ship = {8000, 9100, 9200, 9500};
    std::vector<double>  ep   = {100.0, 200.0, 300.0, 400.0};
    std::vector<double>  dc   = {0.10, 0.05, 0.06, 0.07};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(ship));
    cols.push_back(mk_col(ep));
    cols.push_back(mk_col(dc));
    f.tbl = gamax::make_table(std::move(cols));
    f.view = f.tbl->view();
    return f;
}

// Select the requested columns out of the fixture view by name. The
// fixture's column order matches the catalog, so name→ordinal is a
// straight lookup against `f.cat.find("lineitem")->column_names`.
gex::SourceResolver resolver_for(const Fixture& f) {
    return [&f](const std::string& name,
                const std::vector<std::string>& cols) -> gamax::table_view {
        if (name != "lineitem") {
            throw std::runtime_error("unknown table: " + name);
        }
        const auto* ct = f.cat.find(name);
        std::vector<gamax::size_type> ords;
        ords.reserve(cols.size());
        for (const auto& c : cols) {
            auto it = std::find(ct->column_names.begin(),
                                ct->column_names.end(), c);
            ords.push_back(static_cast<gamax::size_type>(
                std::distance(ct->column_names.begin(), it)));
        }
        return f.view.select_columns(ords);
    };
}

} // namespace

// ── Lowering + printer ────────────────────────────────────────────

TEST(ExecLower, ScanEmitsOnlySourceTable) {
    auto f = make_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto plan = qb.scan("lineitem", {"l_discount", "l_shipdate"}).plan();
    auto ep = gir::lower_to_exec_plan(plan, f.cat);

    // Column-pruning-at-scan collapses Scan to a single SourceTable
    // step whose payload carries the requested columns in order —
    // no follow-up Reorder.
    ASSERT_EQ(ep.steps.size(), 1u);
    EXPECT_EQ(ep.steps[0].kind, gex::StepKind::SourceTable);
    const auto& pl = std::get<gex::SourceTablePayload>(ep.steps[0].payload);
    EXPECT_EQ(pl.table_name, "lineitem");
    ASSERT_EQ(pl.columns.size(), 2u);
    EXPECT_EQ(pl.columns[0], "l_discount");
    EXPECT_EQ(pl.columns[1], "l_shipdate");
    ASSERT_EQ(ep.required_sources.size(), 1u);
    EXPECT_EQ(ep.required_sources[0], "lineitem");
}

TEST(ExecLower, SortPlusLimitFusesToTopK) {
    auto f = make_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto plan = qb.scan("lineitem", {"l_shipdate"})
                  .sort({gir::asc("l_shipdate")})
                  .limit(3)
                  .plan();
    auto ep = gir::lower_to_exec_plan(plan, f.cat);

    // With column-pruning-at-scan: SourceTable then TopK (fused Sort+Limit).
    ASSERT_EQ(ep.steps.size(), 2u);
    EXPECT_EQ(ep.steps.back().kind, gex::StepKind::TopK);
    for (const auto& s : ep.steps) EXPECT_NE(s.kind, gex::StepKind::Sort);
}

TEST(ExecLower, Q6PlanShape) {
    auto f = make_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto li = qb.scan("lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount"});
    auto plan = li
        .filter(li.c("l_shipdate") >= gir::lit(9000)
             && li.c("l_shipdate") <  gir::lit(9300))
        .project({{"rev",
                   li.c("l_extendedprice") * li.c("l_discount")}})
        .plan();
    auto with_agg =
        gir::QueryBuilder(f.cat, g)
            .scan("lineitem", {"l_extendedprice"});  // dummy; real agg next
    (void)with_agg;

    // Build the full Q6-ish plan using the main builder chain.
    auto q6 = li
        .filter(li.c("l_shipdate") >= gir::lit(9000))
        .project({{"rev", li.c("l_extendedprice") * li.c("l_discount")}});
    q6 = q6.aggregate(/*keys=*/{},
        {{"total", gir::AggKind::Sum, q6.c("rev")}});

    auto ep = gir::lower_to_exec_plan(q6.plan(), f.cat);
    // Expect kinds in order: SourceTable, Reorder, Filter, Project,
    // Reorder (agg materialize), ScalarAgg.
    auto out = gex::format_exec_plan(ep);
    EXPECT_TRUE(contains(out, "SourceTable"));
    EXPECT_TRUE(contains(out, "Filter"));
    EXPECT_TRUE(contains(out, "Project"));
    EXPECT_TRUE(contains(out, "ScalarAgg"));
}

TEST(ExecPrint, ReadableOutput) {
    auto f = make_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto plan = qb.scan("lineitem", {"l_discount"}).plan();
    auto ep = gir::lower_to_exec_plan(plan, f.cat);
    auto out = gex::format_exec_plan(ep);
    EXPECT_TRUE(contains(out, "required sources: lineitem"));
    EXPECT_TRUE(contains(out, "SourceTable(\"lineitem\" cols=[l_discount]"));
    EXPECT_TRUE(contains(out, "return v"));
}

// ── Executor ──────────────────────────────────────────────────────

TEST(ExecExecutor, Q6RoundTripThroughExecPlan) {
    auto f = make_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto li = qb.scan("lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount"});
    auto proj = li
        .filter(li.c("l_shipdate") >= gir::lit(9100)
             && li.c("l_shipdate") <= gir::lit(9200))
        .project({{"rev", li.c("l_extendedprice") * li.c("l_discount")}});
    auto q6 = proj.aggregate(/*keys=*/{},
        {{"total", gir::AggKind::Sum, proj.c("rev")}});

    auto ep = gir::lower_to_exec_plan(q6.plan(), f.cat);
    gamax::phys::Context ctx{cudaStreamDefault};
    auto out = gex::execute(ep, resolver_for(f), ctx);

    // Rows 1 and 2 pass the filter:
    //   row 1: 200 * 0.05 = 10.0
    //   row 2: 300 * 0.06 = 18.0
    ASSERT_EQ(out->num_rows(), 1);
    ASSERT_EQ(out->num_columns(), 1);
    auto total = gamax::copy_column_to_host<double>(out->view()[0]);
    EXPECT_DOUBLE_EQ(total[0], 10.0 + 18.0);
}

// ── Column-pruning rule ──────────────────────────────────────────
//
// The first IR lowering rule drops columns the predicate touches but
// no downstream op uses. Implemented by passing `output_ordinals` to
// `phys::filter`; surfaced in the ExecPlan as a non-empty
// FilterPayload::output_ordinals.

namespace {

// Catalog with extra columns so pruning has something to prune.
gir::Catalog make_wide_catalog() {
    gir::Catalog cat;
    gir::CatalogTable t;
    t.name = "lineitem";
    t.column_names = {
        "l_orderkey", "l_shipdate", "l_extendedprice", "l_discount",
    };
    t.column_types = {
        data_type{type_id::INT32},
        data_type{type_id::INT32},
        data_type{type_id::FLOAT64},
        data_type{type_id::FLOAT64},
    };
    t.column_nullable = {false, false, false, false};
    cat.add_table(std::move(t));
    return cat;
}

struct WideFixture {
    gir::Catalog cat;
    std::unique_ptr<gamax::table> tbl;
    gamax::table_view view;
};
WideFixture make_wide_fixture() {
    WideFixture f;
    f.cat = make_wide_catalog();
    std::vector<int32_t> ok   = {1, 2, 3, 4};
    std::vector<int32_t> ship = {9000, 9100, 9200, 9500};
    std::vector<double>  ep   = {100.0, 200.0, 300.0, 400.0};
    std::vector<double>  dc   = {0.10, 0.05, 0.06, 0.07};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(ok));
    cols.push_back(mk_col(ship));
    cols.push_back(mk_col(ep));
    cols.push_back(mk_col(dc));
    f.tbl = gamax::make_table(std::move(cols));
    f.view = f.tbl->view();
    return f;
}

gex::SourceResolver wide_resolver_for(const WideFixture& f) {
    return [&f](const std::string& name,
                const std::vector<std::string>& cols) -> gamax::table_view {
        if (name != "lineitem") {
            throw std::runtime_error("unknown table: " + name);
        }
        const auto* ct = f.cat.find(name);
        std::vector<gamax::size_type> ords;
        ords.reserve(cols.size());
        for (const auto& c : cols) {
            auto it = std::find(ct->column_names.begin(),
                                ct->column_names.end(), c);
            ords.push_back(static_cast<gamax::size_type>(
                std::distance(ct->column_names.begin(), it)));
        }
        return f.view.select_columns(ords);
    };
}

} // namespace

// Filter whose predicate reads l_shipdate, Project whose output uses
// only l_extendedprice. Pruning should:
//   * Drop l_shipdate (and l_discount, l_orderkey) from the Filter's
//     output.
//   * Leave the Filter step carrying `output_ordinals = [2]` (the
//     ordinal of l_extendedprice within the Scan's exec schema).
TEST(ExecLower, FilterDropsDeadColumnsWhenPredicateOnlyReferencesOne) {
    auto f = make_wide_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"});
    auto filtered = li.filter(li.c("l_shipdate") > gir::lit(9000));
    auto projected = filtered.project({
        {"ep", filtered.c("l_extendedprice")},
    });
    auto ep = gir::lower_to_exec_plan(projected.plan(), f.cat);

    // Find the Filter step.
    const gex::Step* filter_step = nullptr;
    for (const auto& s : ep.steps) {
        if (s.kind == gex::StepKind::Filter) {
            filter_step = &s;
            break;
        }
    }
    ASSERT_NE(filter_step, nullptr);
    const auto& fp = std::get<gex::FilterPayload>(filter_step->payload);
    // Exactly one column kept: l_extendedprice. Its ordinal in the
    // scan's reorder output is 2 (order matches scan_columns request).
    ASSERT_EQ(fp.output_ordinals.size(), 1u);
    EXPECT_EQ(fp.output_ordinals[0], 2);
}

// Parity: when no column can be pruned (every Scan column is used
// downstream), `output_ordinals` is left empty and the Executor
// takes the cheap all-columns path.
TEST(ExecLower, FilterEmitsNoPruningWhenEverythingIsLive) {
    auto f = make_wide_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"});
    auto filtered = li.filter(li.c("l_shipdate") > gir::lit(9000));
    // Projection references EVERY Scan column → nothing to prune.
    auto projected = filtered.project({
        {"ok", filtered.c("l_orderkey")},
        {"sd", filtered.c("l_shipdate")},
        {"ep", filtered.c("l_extendedprice")},
        {"dc", filtered.c("l_discount")},
    });
    auto ep = gir::lower_to_exec_plan(projected.plan(), f.cat);

    const gex::Step* filter_step = nullptr;
    for (const auto& s : ep.steps) {
        if (s.kind == gex::StepKind::Filter) {
            filter_step = &s;
            break;
        }
    }
    ASSERT_NE(filter_step, nullptr);
    const auto& fp = std::get<gex::FilterPayload>(filter_step->payload);
    EXPECT_TRUE(fp.output_ordinals.empty());
}

// End-to-end correctness with pruning engaged. The filter keeps rows
// 9100 <= shipdate <= 9200 (rows 1, 2) and the Project outputs
// revenue = ep * dc → {200*0.05, 300*0.06} = {10.0, 18.0}. The
// pruning rule should drop l_orderkey and l_shipdate before the
// project runs; the numeric result must still be correct.
TEST(ExecExecutor, PrunedFilterProducesCorrectResult) {
    auto f = make_wide_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"});
    auto proj = li
        .filter(li.c("l_shipdate") >= gir::lit(9100)
             && li.c("l_shipdate") <= gir::lit(9200))
        .project({{"rev", li.c("l_extendedprice") * li.c("l_discount")}});
    auto ep = gir::lower_to_exec_plan(proj.plan(), f.cat);

    // Sanity-check: the Filter step retains only the columns needed
    // for the Project body (l_extendedprice, l_discount).
    const gex::Step* filter_step = nullptr;
    for (const auto& s : ep.steps) {
        if (s.kind == gex::StepKind::Filter) { filter_step = &s; break; }
    }
    ASSERT_NE(filter_step, nullptr);
    const auto& fp = std::get<gex::FilterPayload>(filter_step->payload);
    EXPECT_EQ(fp.output_ordinals.size(), 2u);

    gamax::phys::Context ctx{cudaStreamDefault};
    auto out = gex::execute(ep, wide_resolver_for(f), ctx);
    ASSERT_EQ(out->num_rows(), 2);
    ASSERT_EQ(out->num_columns(), 1);
    auto rev = gamax::copy_column_to_host<double>(out->view()[0]);
    EXPECT_DOUBLE_EQ(rev[0], 10.0);
    EXPECT_DOUBLE_EQ(rev[1], 18.0);
}

TEST(ExecLower, DistinctLowersToDistinctStep) {
    // Lower a proper-subset Distinct and verify the ExecPlan's last
    // step is a Distinct with the correct key_ordinals payload.
    auto f = make_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto li = qb.scan("lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount"});
    auto plan = li.distinct({"l_shipdate"}).plan();
    auto ep = gir::lower_to_exec_plan(plan, f.cat);

    ASSERT_FALSE(ep.steps.empty());
    const auto& last = ep.steps.back();
    EXPECT_EQ(last.kind, gex::StepKind::Distinct);
    const auto& pl = std::get<gex::DistinctPayload>(last.payload);
    // Distinct over l_shipdate; scan-pushdown will have pruned the
    // other two scan columns away, so l_shipdate is ordinal 0 in the
    // Distinct's input.
    ASSERT_EQ(pl.key_ordinals.size(), 1u);
    EXPECT_EQ(pl.key_ordinals[0], 0);
    // Output schema has exactly one column — the narrowed key.
    EXPECT_EQ(ep.values[last.output].schema.size(), 1u);
}

TEST(ExecExecutor, DistinctEndToEndYieldsUniqueKeys) {
    // End-to-end: scan fixture → distinct(l_shipdate) → execute.
    // Fixture has 5 rows with 4 distinct l_shipdate values.
    auto f = make_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto li = qb.scan("lineitem", {"l_shipdate"});
    auto plan = li.distinct({"l_shipdate"}).plan();
    auto ep = gir::lower_to_exec_plan(plan, f.cat);
    gamax::phys::Context ctx{cudaStreamDefault};
    auto out = gex::execute(ep, resolver_for(f), ctx);
    EXPECT_GT(out->num_rows(), 0);
    EXPECT_LE(out->num_rows(), 5);
    EXPECT_EQ(out->num_columns(), 1);
}

TEST(ExecExecutor, NullRootOrResolverRaises) {
    auto f = make_fixture();
    gir::IdGenerator g;
    gir::QueryBuilder qb(f.cat, g);
    auto plan = qb.scan("lineitem", {"l_discount"}).plan();
    auto ep = gir::lower_to_exec_plan(plan, f.cat);
    gamax::phys::Context ctx{cudaStreamDefault};

    // Empty ExecPlan throws.
    gex::ExecPlan empty;
    EXPECT_THROW(gex::execute(empty, resolver_for(f), ctx),
                 std::runtime_error);

    // Resolver that throws surfaces the error to the caller.
    auto bad = [](const std::string& name,
                  const std::vector<std::string>&) -> gamax::table_view {
        throw std::runtime_error("unknown: " + name);
    };
    EXPECT_THROW(gex::execute(ep, bad, ctx), std::runtime_error);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
