#include <gtest/gtest.h>

#include "ir/catalog.hpp"
#include "ir/expr.hpp"
#include "ir/plan_node.hpp"

namespace gir = gamax::ir;
using gamax::data_type;
using gamax::type_id;

namespace {
data_type t(type_id id) { return data_type{id}; }

gir::Catalog make_tpch_catalog() {
    gir::Catalog cat;
    gir::CatalogTable lineitem;
    lineitem.name = "lineitem";
    lineitem.column_names = {
        "l_shipdate", "l_extendedprice", "l_discount", "l_quantity",
    };
    lineitem.column_types = {
        t(type_id::INT32), t(type_id::FLOAT64),
        t(type_id::FLOAT64), t(type_id::INT32),
    };
    lineitem.column_nullable = {false, false, false, false};
    cat.add_table(std::move(lineitem));

    gir::CatalogTable orders;
    orders.name = "orders";
    orders.column_names  = {"o_orderkey", "o_orderdate", "o_orderpriority"};
    orders.column_types  = {t(type_id::INT64), t(type_id::INT32), t(type_id::STRING)};
    orders.column_nullable = {false, false, false};
    cat.add_table(std::move(orders));
    return cat;
}

gir::PlanPtr scan_orders(
    const gir::Catalog& cat, gir::IdGenerator& gen)
{
    return gir::scan(cat, "orders",
        {"o_orderkey", "o_orderdate", "o_orderpriority"}, gen);
}

gir::Expr field_ref(const gir::Schema& s, size_t i) {
    return gir::col_ref(s[i].id, s[i].type, s[i].nullable);
}
} // namespace

TEST(IrPlanNode, ScanBuildsSchemaFromCatalog) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    ASSERT_EQ(s->kind, gir::NodeKind::Scan);
    ASSERT_EQ(s->output_schema.size(), 2u);
    EXPECT_EQ(s->output_schema[0].name, "l_shipdate");
    EXPECT_EQ(s->output_schema[0].type, t(type_id::INT32));
    EXPECT_EQ(s->output_schema[1].name, "l_discount");
    EXPECT_EQ(s->output_schema[1].type, t(type_id::FLOAT64));
    EXPECT_NE(s->output_schema[0].id, s->output_schema[1].id);
}

TEST(IrPlanNode, ScanMissingTableThrows) {
    gir::Catalog cat;
    EXPECT_THROW(gir::scan(cat, "nope", {"c"}), std::runtime_error);
}

TEST(IrPlanNode, ScanMissingColumnThrows) {
    auto cat = make_tpch_catalog();
    EXPECT_THROW(gir::scan(cat, "lineitem", {"no_such_column"}),
                 std::runtime_error);
}

TEST(IrPlanNode, ScanEmptyColumnsThrows) {
    auto cat = make_tpch_catalog();
    EXPECT_THROW(gir::scan(cat, "lineitem", {}), std::runtime_error);
}

TEST(IrPlanNode, ScanEachRunAssignsFreshIds) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s1 = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto s2 = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    EXPECT_NE(s1->output_schema[0].id, s2->output_schema[0].id);
}

TEST(IrPlanNode, FilterPreservesSchemaIdentity) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    auto pred = field_ref(s->output_schema, 0) < gir::lit(10000);
    auto f = gir::filter(s, std::move(pred));
    ASSERT_EQ(f->kind, gir::NodeKind::Filter);
    ASSERT_EQ(f->output_schema.size(), 2u);
    EXPECT_EQ(f->output_schema[0].id, s->output_schema[0].id);
    EXPECT_EQ(f->output_schema[1].id, s->output_schema[1].id);
    ASSERT_EQ(f->inputs.size(), 1u);
    EXPECT_EQ(f->inputs[0].get(), s.get());
}

TEST(IrPlanNode, FilterRejectsNonBool) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto not_bool = field_ref(s->output_schema, 0) + gir::lit(1);
    EXPECT_THROW(gir::filter(s, std::move(not_bool)), std::runtime_error);
}

TEST(IrPlanNode, FilterRejectsUnknownColumn) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    // A ColumnId minted outside any scan — not in s's schema.
    auto bogus = gir::col_ref(gen.next(), t(type_id::BOOL8), false);
    EXPECT_THROW(gir::filter(s, std::move(bogus)), std::runtime_error);
}

TEST(IrPlanNode, ProjectPureColRefPreservesId) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    auto discount_id = s->output_schema[1].id;

    std::vector<gir::ProjectItem> items;
    items.push_back({"d", field_ref(s->output_schema, 1)});
    auto p = gir::project(s, std::move(items), gen);

    ASSERT_EQ(p->output_schema.size(), 1u);
    EXPECT_EQ(p->output_schema[0].id, discount_id);
    EXPECT_EQ(p->output_schema[0].name, "d");
}

TEST(IrPlanNode, ProjectComputedExprGetsFreshId) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_extendedprice", "l_discount"}, gen);

    std::vector<gir::ProjectItem> items;
    items.push_back({"revenue",
        field_ref(s->output_schema, 0) * field_ref(s->output_schema, 1)});
    auto p = gir::project(s, std::move(items), gen);

    ASSERT_EQ(p->output_schema.size(), 1u);
    EXPECT_NE(p->output_schema[0].id, s->output_schema[0].id);
    EXPECT_NE(p->output_schema[0].id, s->output_schema[1].id);
    EXPECT_EQ(p->output_schema[0].type, t(type_id::FLOAT64));
    EXPECT_EQ(p->output_schema[0].name, "revenue");
}

TEST(IrPlanNode, ProjectRejectsUnknownColumn) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    std::vector<gir::ProjectItem> items;
    items.push_back({"x", gir::col_ref(gen.next(), t(type_id::INT32), false)});
    EXPECT_THROW(gir::project(s, std::move(items), gen), std::runtime_error);
}

TEST(IrPlanNode, ScalarAggregate) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_extendedprice"}, gen);

    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"total", gir::AggKind::Sum, field_ref(s->output_schema, 0)});
    auto a = gir::aggregate(s, /*keys=*/{}, std::move(aggs), gen);

    ASSERT_EQ(a->output_schema.size(), 1u);
    EXPECT_EQ(a->output_schema[0].type, t(type_id::FLOAT64));
    EXPECT_EQ(a->output_schema[0].name, "total");
    EXPECT_TRUE(a->output_schema[0].nullable);  // SUM over empty -> NULL
}

TEST(IrPlanNode, GroupedAggregate) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate", "l_extendedprice"}, gen);
    auto shipdate_id = s->output_schema[0].id;

    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"cnt", gir::AggKind::Count, field_ref(s->output_schema, 1)});
    auto a = gir::aggregate(s, /*keys=*/{shipdate_id}, std::move(aggs), gen);

    ASSERT_EQ(a->output_schema.size(), 2u);
    EXPECT_EQ(a->output_schema[0].id, shipdate_id);  // key id preserved
    EXPECT_EQ(a->output_schema[0].name, "l_shipdate");
    EXPECT_EQ(a->output_schema[1].type, t(type_id::INT64));
    EXPECT_EQ(a->output_schema[1].name, "cnt");
    EXPECT_FALSE(a->output_schema[1].nullable);  // COUNT is never NULL
}

TEST(IrPlanNode, AggregateAvgIsFloat64) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_quantity"}, gen);
    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"avg_q", gir::AggKind::Avg, field_ref(s->output_schema, 0)});
    auto a = gir::aggregate(s, /*keys=*/{}, std::move(aggs), gen);
    EXPECT_EQ(a->output_schema[0].type, t(type_id::FLOAT64));
}

TEST(IrPlanNode, AggregateMinPreservesType) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"min_d", gir::AggKind::Min, field_ref(s->output_schema, 0)});
    auto a = gir::aggregate(s, /*keys=*/{}, std::move(aggs), gen);
    EXPECT_EQ(a->output_schema[0].type, t(type_id::INT32));
}

TEST(IrPlanNode, AggregateCountAllIgnoresBody) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"n", gir::AggKind::CountAll, gir::lit(int64_t{0})});
    auto a = gir::aggregate(s, /*keys=*/{}, std::move(aggs), gen);
    EXPECT_EQ(a->output_schema[0].type, t(type_id::INT64));
    EXPECT_FALSE(a->output_schema[0].nullable);
}

TEST(IrPlanNode, AggregateRejectsUnknownKey) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    gir::ColumnId fake;
    fake.value = 99999;
    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"n", gir::AggKind::CountAll, gir::lit(int64_t{0})});
    EXPECT_THROW(gir::aggregate(s, {fake}, std::move(aggs), gen),
                 std::runtime_error);
}

TEST(IrPlanNode, SortPreservesSchema) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    std::vector<gir::SortKey> keys = {
        {s->output_schema[0].id, gamax::sort_order::ASCENDING},
        {s->output_schema[1].id, gamax::sort_order::DESCENDING},
    };
    auto sorted = gir::sort(s, std::move(keys));
    ASSERT_EQ(sorted->kind, gir::NodeKind::Sort);
    ASSERT_EQ(sorted->output_schema.size(), 2u);
    EXPECT_EQ(sorted->output_schema[0].id, s->output_schema[0].id);
    EXPECT_EQ(sorted->sort_keys.size(), 2u);
    EXPECT_EQ(sorted->sort_keys[1].order, gamax::sort_order::DESCENDING);
}

TEST(IrPlanNode, SortRejectsUnknownKey) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    gir::ColumnId bogus;
    bogus.value = 99999;
    std::vector<gir::SortKey> keys = {{bogus, gamax::sort_order::ASCENDING}};
    EXPECT_THROW(gir::sort(s, std::move(keys)), std::runtime_error);
}

TEST(IrPlanNode, LimitPreservesSchema) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto l = gir::limit(s, 10);
    ASSERT_EQ(l->kind, gir::NodeKind::Limit);
    ASSERT_EQ(l->output_schema.size(), 1u);
    EXPECT_EQ(l->limit_n, 10);
    EXPECT_EQ(l->limit_offset, 0);
    EXPECT_EQ(l->output_schema[0].id, s->output_schema[0].id);
}

TEST(IrPlanNode, LimitRejectsNegative) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    EXPECT_THROW(gir::limit(s, -1), std::runtime_error);
    EXPECT_THROW(gir::limit(s, 5, -1), std::runtime_error);
}

TEST(IrPlanNode, InnerJoinConcatenatesSchemas) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto li = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto o  = scan_orders(cat, gen);

    auto on = field_ref(li->output_schema, 0) == field_ref(o->output_schema, 1);
    auto j = gir::inner_join(li, o, std::move(on));
    ASSERT_EQ(j->kind, gir::NodeKind::Join);
    EXPECT_EQ(j->join_kind, gir::JoinKind::Inner);
    // 1 (lineitem) + 3 (orders) = 4 output columns
    ASSERT_EQ(j->output_schema.size(), 4u);
    EXPECT_EQ(j->output_schema[0].id, li->output_schema[0].id);
    EXPECT_EQ(j->output_schema[3].id, o->output_schema[2].id);
}

TEST(IrPlanNode, InnerJoinRejectsNonBoolCondition) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto li = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto o  = scan_orders(cat, gen);
    auto not_bool = field_ref(li->output_schema, 0) + gir::lit(1);
    EXPECT_THROW(gir::inner_join(li, o, std::move(not_bool)),
                 std::runtime_error);
}

TEST(IrPlanNode, InnerJoinRejectsRefOutsideEitherSide) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto li = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto o  = scan_orders(cat, gen);
    // A column from an unrelated third scan.
    auto extra = gir::scan(cat, "lineitem", {"l_discount"}, gen);
    auto bad_on = field_ref(extra->output_schema, 0) >= gir::lit(0.0);
    EXPECT_THROW(gir::inner_join(li, o, std::move(bad_on)),
                 std::runtime_error);
}

TEST(IrPlanNode, LeftOuterJoinMakesRightSideNullable) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto li = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto o  = scan_orders(cat, gen);

    auto on = field_ref(li->output_schema, 0) == field_ref(o->output_schema, 1);
    auto j = gir::left_outer_join(li, o, std::move(on));
    ASSERT_EQ(j->output_schema.size(), 4u);
    EXPECT_FALSE(j->output_schema[0].nullable);  // left side stays as-is
    EXPECT_TRUE(j->output_schema[1].nullable);   // right side is NULL-extended
    EXPECT_TRUE(j->output_schema[3].nullable);
}

TEST(IrPlanNode, LeftSemiJoinKeepsOnlyLeftSchema) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto li = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    auto o  = scan_orders(cat, gen);

    auto on = field_ref(li->output_schema, 0) == field_ref(o->output_schema, 1);
    auto j = gir::left_semi_join(li, o, std::move(on));
    ASSERT_EQ(j->output_schema.size(), 2u);
    EXPECT_EQ(j->output_schema[0].id, li->output_schema[0].id);
    EXPECT_EQ(j->output_schema[1].id, li->output_schema[1].id);
}

TEST(IrPlanNode, LeftAntiJoinKeepsOnlyLeftSchema) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto li = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto o  = scan_orders(cat, gen);

    auto on = field_ref(li->output_schema, 0) == field_ref(o->output_schema, 1);
    auto j = gir::left_anti_join(li, o, std::move(on));
    ASSERT_EQ(j->output_schema.size(), 1u);
    EXPECT_EQ(j->join_kind, gir::JoinKind::LeftAnti);
    EXPECT_EQ(j->output_schema[0].id, li->output_schema[0].id);
}

TEST(IrPlanNode, CrossJoinConcatenatesSchemasNoCondition) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto li = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto o  = scan_orders(cat, gen);

    auto j = gir::cross_join(li, o);
    ASSERT_EQ(j->output_schema.size(), 4u);
    EXPECT_EQ(j->join_kind, gir::JoinKind::Cross);
    EXPECT_FALSE(j->join_condition.has_value());
}

TEST(IrPlanNode, DistinctPreservesSchema) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    auto d = gir::distinct(s);  // all-columns distinct
    ASSERT_EQ(d->kind, gir::NodeKind::Distinct);
    ASSERT_EQ(d->output_schema.size(), 2u);
    EXPECT_TRUE(d->distinct_keys.empty());
}

TEST(IrPlanNode, DistinctKeysCoveringAllInputsPreservesSchema) {
    // Explicit keys that cover every input column (by id) collapse to
    // the same shape as the empty-keys case: output_schema == input.
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    auto d = gir::distinct(s,
        {s->output_schema[0].id, s->output_schema[1].id});
    ASSERT_EQ(d->output_schema.size(), 2u);
    EXPECT_EQ(d->output_schema[0].id, s->output_schema[0].id);
    EXPECT_EQ(d->output_schema[1].id, s->output_schema[1].id);
    EXPECT_EQ(d->distinct_keys.size(), 2u);
}

TEST(IrPlanNode, DistinctProperSubsetNarrowsSchemaToKeysInCallerOrder) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount"}, gen);
    // Pick 2-of-3 keys in non-input order. Output should contain
    // exactly those two columns, in the caller's order.
    auto d = gir::distinct(s,
        {s->output_schema[2].id, s->output_schema[0].id});
    ASSERT_EQ(d->output_schema.size(), 2u);
    EXPECT_EQ(d->output_schema[0].id, s->output_schema[2].id);
    EXPECT_EQ(d->output_schema[1].id, s->output_schema[0].id);
    EXPECT_EQ(d->output_schema[0].name, "l_discount");
    EXPECT_EQ(d->output_schema[1].name, "l_shipdate");
}

// ── Concat ───────────────────────────────────────────────────────

TEST(IrPlanNode, ConcatRejectsEmptyInputsVector) {
    std::vector<gir::PlanPtr> empty;
    EXPECT_THROW(gir::concat(std::move(empty)), std::runtime_error);
}

TEST(IrPlanNode, ConcatSingleInputIsIdentityInShape) {
    // 1-input concat is legal; the output schema matches the sole
    // input's, modulo fresh ColumnIds.
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    auto c = gir::concat({s}, gen);
    ASSERT_EQ(c->kind, gir::NodeKind::Concat);
    ASSERT_EQ(c->output_schema.size(), 2u);
    EXPECT_EQ(c->output_schema[0].type, t(type_id::INT32));
    EXPECT_EQ(c->output_schema[1].type, t(type_id::FLOAT64));
    // Fresh ids — must differ from the Scan's ids.
    EXPECT_NE(c->output_schema[0].id, s->output_schema[0].id);
    EXPECT_NE(c->output_schema[1].id, s->output_schema[1].id);
}

TEST(IrPlanNode, ConcatBinaryTakesNamesFromFirstInput) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto a = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    auto b = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    auto c = gir::concat({a, b}, gen);
    ASSERT_EQ(c->output_schema.size(), 2u);
    // Same names as the first input.
    EXPECT_EQ(c->output_schema[0].name, "l_shipdate");
    EXPECT_EQ(c->output_schema[1].name, "l_discount");
    // Fresh ids differ from both inputs.
    EXPECT_NE(c->output_schema[0].id, a->output_schema[0].id);
    EXPECT_NE(c->output_schema[0].id, b->output_schema[0].id);
}

TEST(IrPlanNode, ConcatRejectsArityMismatch) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto a = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto b = gir::scan(cat, "lineitem", {"l_shipdate", "l_discount"}, gen);
    EXPECT_THROW(gir::concat({a, b}, gen), std::runtime_error);
}

TEST(IrPlanNode, ConcatRejectsTypeMismatch) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto a = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);   // INT32
    auto b = gir::scan(cat, "lineitem", {"l_discount"}, gen);   // FLOAT64
    EXPECT_THROW(gir::concat({a, b}, gen), std::runtime_error);
}

TEST(IrPlanNode, ConcatThreeWayWorks) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto a = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto b = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto d = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    auto c = gir::concat({a, b, d}, gen);
    ASSERT_EQ(c->output_schema.size(), 1u);
    ASSERT_EQ(c->inputs.size(), 3u);
}

TEST(IrPlanNode, DistinctRejectsUnknownKey) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_shipdate"}, gen);
    gir::ColumnId bogus;
    bogus.value = 42;
    EXPECT_THROW(gir::distinct(s, {bogus}), std::runtime_error);
}

TEST(IrPlanNode, AggregateCountDistinctIsInt64) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto s = gir::scan(cat, "lineitem", {"l_quantity"}, gen);
    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"ndv", gir::AggKind::CountDistinct,
                    field_ref(s->output_schema, 0)});
    auto a = gir::aggregate(s, /*keys=*/{}, std::move(aggs), gen);
    EXPECT_EQ(a->output_schema[0].type, t(type_id::INT64));
    EXPECT_FALSE(a->output_schema[0].nullable);
}

// End-to-end: build Q6's plan (scan -> filter -> project -> scalar agg)
// and check the final output schema.
TEST(IrPlanNode, Q6PlanShape) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();

    auto li = gir::scan(cat, "lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount", "l_quantity"}, gen);
    ASSERT_EQ(li->output_schema.size(), 4u);

    auto predicate =
        (field_ref(li->output_schema, 0) >= gir::lit(8766))
     && (field_ref(li->output_schema, 0) <  gir::lit(9131))
     && (field_ref(li->output_schema, 2) >= gir::lit(0.05))
     && (field_ref(li->output_schema, 2) <= gir::lit(0.07))
     && (field_ref(li->output_schema, 3) <  gir::lit(24));
    auto f = gir::filter(li, std::move(predicate));

    std::vector<gir::ProjectItem> proj;
    proj.push_back({"revenue",
        field_ref(li->output_schema, 1) * field_ref(li->output_schema, 2)});
    auto p = gir::project(f, std::move(proj), gen);
    ASSERT_EQ(p->output_schema.size(), 1u);

    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"total", gir::AggKind::Sum, field_ref(p->output_schema, 0)});
    auto a = gir::aggregate(p, /*keys=*/{}, std::move(aggs), gen);

    ASSERT_EQ(a->output_schema.size(), 1u);
    EXPECT_EQ(a->output_schema[0].name, "total");
    EXPECT_EQ(a->output_schema[0].type, t(type_id::FLOAT64));
}

// End-to-end: join + grouped aggregate + sort + limit, i.e. the shape
// of Q3 / Q10 / Q18. Exercises the operators added for full TPC-H
// coverage.
TEST(IrPlanNode, JoinAggregateSortLimitShape) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    auto li = gir::scan(cat, "lineitem", {"l_shipdate", "l_extendedprice"}, gen);
    auto o  = scan_orders(cat, gen);

    // Join on l_shipdate == o_orderdate (contrived, but exercises the
    // join validation path).
    auto on = field_ref(li->output_schema, 0) == field_ref(o->output_schema, 1);
    auto j = gir::inner_join(li, o, std::move(on));
    ASSERT_EQ(j->output_schema.size(), 5u);

    // Group by o_orderpriority, SUM(l_extendedprice).
    auto priority_id = j->output_schema[4].id;  // o_orderpriority
    auto price_ref   = field_ref(j->output_schema, 1);  // l_extendedprice
    std::vector<gir::AggregateItem> aggs;
    aggs.push_back({"revenue", gir::AggKind::Sum, std::move(price_ref)});
    auto a = gir::aggregate(j, {priority_id}, std::move(aggs), gen);

    // ORDER BY revenue DESC
    auto revenue_id = a->output_schema[1].id;
    std::vector<gir::SortKey> keys = {
        {revenue_id, gamax::sort_order::DESCENDING},
    };
    auto sorted = gir::sort(a, std::move(keys));

    // LIMIT 10
    auto top = gir::limit(sorted, 10);
    ASSERT_EQ(top->kind, gir::NodeKind::Limit);
    ASSERT_EQ(top->output_schema.size(), 2u);
    EXPECT_EQ(top->output_schema[0].id, priority_id);
    EXPECT_EQ(top->output_schema[1].id, revenue_id);
    EXPECT_EQ(top->limit_n, 10);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
