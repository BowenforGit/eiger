#include <gtest/gtest.h>

#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/liveness.hpp"

namespace gir = gamax::ir;
using gamax::data_type;
using gamax::type_id;

namespace {

gir::Catalog make_catalog() {
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

    gir::CatalogTable o;
    o.name = "orders";
    o.column_names = {"o_orderkey", "o_orderdate"};
    o.column_types = {
        data_type{type_id::INT32},
        data_type{type_id::INT32},
    };
    o.column_nullable = {false, false};
    cat.add_table(std::move(o));
    return cat;
}

// True iff every ColumnId in the node's output_schema is present in
// `live_out[node]`.
bool all_output_columns_live(
    const gir::LivenessResult& res, const gir::PlanNode& node)
{
    const auto& s = res.at(&node);
    for (const auto& f : node.output_schema.fields()) {
        if (!s.count(f.id)) return false;
    }
    return true;
}

} // namespace

TEST(Liveness, RootHasAllOutputLive) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"});
    auto res = gir::compute_liveness(*li.plan());
    EXPECT_TRUE(all_output_columns_live(res, *li.plan()));
}

// The core case: a filter's predicate references a column that no
// downstream op consumes. The child's liveness must still include
// that column (Filter needs it to evaluate the predicate), but the
// Filter's OWN live_out must not.
TEST(Liveness, FilterPredicateOnlyColumnNotLiveAtFilterOutput) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"});
    auto filtered = li.filter(li.c("l_shipdate") > gir::lit(9100));
    auto projected = filtered.project({
        {"k", filtered.c("l_orderkey")},
        {"p", filtered.c("l_extendedprice")},
    });

    auto res = gir::compute_liveness(*projected.plan());

    // Filter's output is whatever Project needs from it: l_orderkey +
    // l_extendedprice. l_shipdate (predicate-only) and l_discount
    // (unused) should NOT be in live_out[filter].
    const auto* filter_node = filtered.plan().get();
    const auto& filter_live = res.at(filter_node);

    auto id_of = [&](const std::string& name) {
        auto idx = filtered.schema().index_of(name);
        return filtered.schema()[*idx].id;
    };
    EXPECT_TRUE(filter_live.count(id_of("l_orderkey")));
    EXPECT_TRUE(filter_live.count(id_of("l_extendedprice")));
    EXPECT_FALSE(filter_live.count(id_of("l_shipdate")));
    EXPECT_FALSE(filter_live.count(id_of("l_discount")));

    // The Scan (filter's child) must still have l_shipdate live —
    // the filter predicate reads it. l_discount is dead end-to-end.
    const auto* scan_node = li.plan().get();
    const auto& scan_live = res.at(scan_node);
    EXPECT_TRUE(scan_live.count(id_of("l_shipdate")));
    EXPECT_TRUE(scan_live.count(id_of("l_orderkey")));
    EXPECT_TRUE(scan_live.count(id_of("l_extendedprice")));
    EXPECT_FALSE(scan_live.count(id_of("l_discount")));
}

// Sort-key columns stay live past the Sort (and into its child).
TEST(Liveness, SortKeysStayLiveAtChildOutput) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_shipdate", "l_extendedprice"});
    auto sorted = li.sort({{"l_shipdate", gamax::sort_order::ASCENDING}});
    auto final_ = sorted.project({
        {"k", sorted.c("l_orderkey")},
        {"p", sorted.c("l_extendedprice")},
    });

    auto res = gir::compute_liveness(*final_.plan());

    auto id_of = [&](const std::string& name) {
        auto idx = li.schema().index_of(name);
        return li.schema()[*idx].id;
    };
    // Sort's live_out = what Project needs = {l_orderkey, l_extendedprice}.
    const auto& sort_live = res.at(sorted.plan().get());
    EXPECT_FALSE(sort_live.count(id_of("l_shipdate")));
    // Scan (= Sort's child) must keep l_shipdate live so the Sort can
    // actually use it as a key.
    const auto& scan_live = res.at(li.plan().get());
    EXPECT_TRUE(scan_live.count(id_of("l_shipdate")));
    EXPECT_TRUE(scan_live.count(id_of("l_orderkey")));
    EXPECT_TRUE(scan_live.count(id_of("l_extendedprice")));
}

// Join: each side only needs columns referenced either by the join
// condition on that side, or downstream from the join.
TEST(Liveness, JoinPropagatesToEachSideIndependently) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"});
    auto o = qb.scan("orders", {"o_orderkey", "o_orderdate"});

    auto joined = li.inner_join(o,
        li.c("l_orderkey") == o.c("o_orderkey"));
    auto projected = joined.project({
        {"ep", joined.c("l_extendedprice")},
        {"od", joined.c("o_orderdate")},
    });

    auto res = gir::compute_liveness(*projected.plan());

    auto li_id = [&](const std::string& n) {
        return li.schema()[*li.schema().index_of(n)].id;
    };
    auto o_id = [&](const std::string& n) {
        return o.schema()[*o.schema().index_of(n)].id;
    };

    const auto& left_live  = res.at(li.plan().get());
    const auto& right_live = res.at(o.plan().get());

    // Left: l_orderkey (join cond) + l_extendedprice (Project). Not
    // l_shipdate / l_discount.
    EXPECT_TRUE(left_live.count(li_id("l_orderkey")));
    EXPECT_TRUE(left_live.count(li_id("l_extendedprice")));
    EXPECT_FALSE(left_live.count(li_id("l_shipdate")));
    EXPECT_FALSE(left_live.count(li_id("l_discount")));

    // Right: o_orderkey (join cond) + o_orderdate (Project).
    EXPECT_TRUE(right_live.count(o_id("o_orderkey")));
    EXPECT_TRUE(right_live.count(o_id("o_orderdate")));
}

// Aggregate: keys are always pulled from child (they drive grouping),
// even if the output key column isn't live at the aggregate's output.
TEST(Liveness, AggregateKeysAlwaysLiveAtChild) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"});
    auto agg = li.aggregate(
        {"l_orderkey"},
        {{"total", gir::AggKind::Sum, li.c("l_extendedprice")}});
    auto projected = agg.project({
        {"t", agg.c("total")},   // drop the key column from Project's output
    });

    auto res = gir::compute_liveness(*projected.plan());

    auto id_of = [&](const std::string& n) {
        return li.schema()[*li.schema().index_of(n)].id;
    };
    const auto& scan_live = res.at(li.plan().get());
    EXPECT_TRUE(scan_live.count(id_of("l_orderkey")));  // grouping key
    EXPECT_TRUE(scan_live.count(id_of("l_extendedprice")));
    EXPECT_FALSE(scan_live.count(id_of("l_shipdate")));
    EXPECT_FALSE(scan_live.count(id_of("l_discount")));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
