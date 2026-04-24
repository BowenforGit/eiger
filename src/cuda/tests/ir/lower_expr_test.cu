#include <gtest/gtest.h>

#include "ir/expr.hpp"
#include "ir/lower_expr.hpp"
#include "ir/schema.hpp"

namespace gir = gamax::ir;
using gamax::data_type;
using gamax::type_id;
using gamax::expression::ExprId;

namespace {
data_type t(type_id id) { return data_type{id}; }

// Builds a small schema with a few columns of specified types, in order.
// Returns the Schema and a vector of the created ColumnIds (same index
// ordering as `types`) so tests can reach in and reference them.
struct BuiltSchema {
    gir::Schema schema;
    std::vector<gir::ColumnId> ids;
};
BuiltSchema make_schema(std::initializer_list<gamax::data_type> types) {
    gir::IdGenerator gen;
    BuiltSchema b;
    for (auto ty : types) {
        gir::Field f;
        f.id = gen.next();
        f.type = ty;
        f.nullable = false;
        b.ids.push_back(f.id);
        b.schema.append(std::move(f));
    }
    return b;
}
} // namespace

TEST(IrLowerExpr, ResolveOrdinal) {
    auto b = make_schema({t(type_id::INT32), t(type_id::FLOAT64)});
    EXPECT_EQ(gir::resolve_ordinal(b.schema, b.ids[0]), 0);
    EXPECT_EQ(gir::resolve_ordinal(b.schema, b.ids[1]), 1);
    gir::ColumnId missing;
    missing.value = 9999;
    EXPECT_THROW(gir::resolve_ordinal(b.schema, missing), std::runtime_error);
}

TEST(IrLowerExpr, ContainsStringOp) {
    auto b = make_schema({t(type_id::STRING), t(type_id::INT32)});
    auto s = gir::col_ref(b.ids[0], t(type_id::STRING), false);
    auto i = gir::col_ref(b.ids[1], t(type_id::INT32), false);

    auto str = gir::string_op(gir::StringOpKind::Equal, std::move(s), {"foo"});
    auto arith = i + gir::lit(1);
    EXPECT_TRUE(gir::expr_contains_string_op(str));
    EXPECT_FALSE(gir::expr_contains_string_op(arith));

    // A compound with a string op somewhere in it
    auto s2 = gir::col_ref(b.ids[0], t(type_id::STRING), false);
    auto mixed =
        gir::string_op(gir::StringOpKind::Startswith, std::move(s2), {"P"})
     && (i > gir::lit(0));
    EXPECT_TRUE(gir::expr_contains_string_op(mixed));
}

TEST(IrLowerExpr, ColRefLoweredToColumnLeaf) {
    auto b = make_schema({t(type_id::INT32), t(type_id::FLOAT64)});
    auto e = gir::col_ref(b.ids[1], t(type_id::FLOAT64), false);
    auto tree = gir::lower_expr_to_tree(e, b.schema);
    EXPECT_EQ(tree.root().id, ExprId::Column);
    EXPECT_EQ(tree.root().data.col.cid, 1);  // ordinal 1
    EXPECT_EQ(tree.root().type, t(type_id::FLOAT64));
}

TEST(IrLowerExpr, LiteralLoweredToConstant) {
    gir::Schema empty;
    auto e = gir::lit(int64_t{7});
    auto tree = gir::lower_expr_to_tree(e, empty);
    EXPECT_EQ(tree.root().id, ExprId::Constant);
    EXPECT_EQ(tree.root().type, t(type_id::INT64));
    EXPECT_EQ(tree.root().data.c.value.get<int64_t>(), 7);
}

TEST(IrLowerExpr, ArithmeticChain) {
    auto b = make_schema({t(type_id::FLOAT64), t(type_id::FLOAT64)});
    auto a = gir::col_ref(b.ids[0], t(type_id::FLOAT64), false);
    auto d = gir::col_ref(b.ids[1], t(type_id::FLOAT64), false);
    auto e = a * d;
    auto tree = gir::lower_expr_to_tree(e, b.schema);
    EXPECT_EQ(tree.root().id, ExprId::Multiply);
    EXPECT_EQ(tree.root().type, t(type_id::FLOAT64));
    ASSERT_EQ(tree.children().size(), 2u);
    EXPECT_EQ(tree.children()[0]->root().id, ExprId::Column);
    EXPECT_EQ(tree.children()[1]->root().id, ExprId::Column);
}

TEST(IrLowerExpr, CompareLowering) {
    auto b = make_schema({t(type_id::INT32)});
    auto c = gir::col_ref(b.ids[0], t(type_id::INT32), false);
    auto ge = c >= gir::lit(10);
    auto lt = c <  gir::lit(10);
    auto eq = c == gir::lit(10);
    EXPECT_EQ(gir::lower_expr_to_tree(ge, b.schema).root().id,
              ExprId::GreaterEqual);
    EXPECT_EQ(gir::lower_expr_to_tree(lt, b.schema).root().id,
              ExprId::LessThan);
    EXPECT_EQ(gir::lower_expr_to_tree(eq, b.schema).root().id,
              ExprId::Equal);
}

TEST(IrLowerExpr, NeDoesNotLowerIntoTree) {
    auto b = make_schema({t(type_id::INT32)});
    auto c = gir::col_ref(b.ids[0], t(type_id::INT32), false);
    auto ne = c != gir::lit(10);
    EXPECT_THROW(gir::lower_expr_to_tree(ne, b.schema), std::runtime_error);
}

TEST(IrLowerExpr, UnaryNotDoesNotLowerIntoTree) {
    auto b = make_schema({t(type_id::INT32)});
    auto c = gir::col_ref(b.ids[0], t(type_id::INT32), false);
    auto not_ge = !(c >= gir::lit(0));
    EXPECT_THROW(gir::lower_expr_to_tree(not_ge, b.schema), std::runtime_error);
}

TEST(IrLowerExpr, LogicalAndOrInTree) {
    auto b = make_schema({t(type_id::INT32)});
    auto c = gir::col_ref(b.ids[0], t(type_id::INT32), false);
    auto e = (c >= gir::lit(1)) && (c < gir::lit(10));
    auto tree = gir::lower_expr_to_tree(e, b.schema);
    EXPECT_EQ(tree.root().id, ExprId::And);
    EXPECT_EQ(tree.root().type, t(type_id::BOOL8));
}

TEST(IrLowerExpr, CaseLowersToSelect) {
    auto b = make_schema({t(type_id::BOOL8), t(type_id::FLOAT64)});
    auto cond = gir::col_ref(b.ids[0], t(type_id::BOOL8), false);
    auto v    = gir::col_ref(b.ids[1], t(type_id::FLOAT64), false);
    auto e = gir::case_when(std::move(cond), std::move(v), gir::lit(0.0));
    auto tree = gir::lower_expr_to_tree(e, b.schema);
    EXPECT_EQ(tree.root().id, ExprId::Select);
    ASSERT_EQ(tree.children().size(), 3u);
}

// ── Predicate lowering ──

TEST(IrLowerExpr, PurelyNumericFilterFoldsIntoOnePredicateExpr) {
    auto b = make_schema({
        t(type_id::INT32), t(type_id::FLOAT64),
        t(type_id::FLOAT64), t(type_id::INT32),
    });
    auto shipdate = gir::col_ref(b.ids[0], t(type_id::INT32), false);
    auto discount = gir::col_ref(b.ids[2], t(type_id::FLOAT64), false);
    auto qty      = gir::col_ref(b.ids[3], t(type_id::INT32), false);
    auto filter =
        (shipdate >= gir::lit(8766))
     && (shipdate <  gir::lit(9131))
     && (discount >= gir::lit(0.05))
     && (discount <= gir::lit(0.07))
     && (qty      <  gir::lit(24));

    auto p = gir::lower_expr_to_predicate(filter, b.schema);
    // Pure numeric => a single Predicate::expr leaf wrapping one
    // ExprTree (no Predicate::AND decomposition).
    EXPECT_EQ(p.type(), gamax::select::PredicateNodeType::EXPR);
    EXPECT_EQ(p.expr_tree().root().id, ExprId::And);
}

TEST(IrLowerExpr, NeInPredicateBecomesNotEq) {
    auto b = make_schema({t(type_id::INT32)});
    auto c = gir::col_ref(b.ids[0], t(type_id::INT32), false);
    auto p = gir::lower_expr_to_predicate(c != gir::lit(10), b.schema);
    EXPECT_EQ(p.type(), gamax::select::PredicateNodeType::NOT);
    ASSERT_EQ(p.children().size(), 1u);
    EXPECT_EQ(p.children()[0]->type(),
              gamax::select::PredicateNodeType::EXPR);
    EXPECT_EQ(p.children()[0]->expr_tree().root().id, ExprId::Equal);
}

TEST(IrLowerExpr, MixedStringAndNumericDecomposesAtPredicateLevel) {
    auto b = make_schema({t(type_id::STRING), t(type_id::INT32)});
    auto s = gir::col_ref(b.ids[0], t(type_id::STRING), false);
    auto i = gir::col_ref(b.ids[1], t(type_id::INT32), false);
    auto filter =
        gir::string_op(gir::StringOpKind::Startswith, std::move(s), {"PROMO"})
     && (i >= gir::lit(10));

    auto p = gir::lower_expr_to_predicate(filter, b.schema);
    EXPECT_EQ(p.type(), gamax::select::PredicateNodeType::AND);
    ASSERT_EQ(p.children().size(), 2u);
    EXPECT_EQ(p.children()[0]->type(),
              gamax::select::PredicateNodeType::STRING_STARTSWITH);
    EXPECT_EQ(p.children()[1]->type(),
              gamax::select::PredicateNodeType::EXPR);
    // The numeric half collapses to a single Predicate::expr with a
    // GreaterEqual at its root.
    EXPECT_EQ(p.children()[1]->expr_tree().root().id, ExprId::GreaterEqual);
}

TEST(IrLowerExpr, StringOpLeavesReferenceOrdinal) {
    auto b = make_schema({t(type_id::INT32), t(type_id::STRING)});
    auto s = gir::col_ref(b.ids[1], t(type_id::STRING), false);
    auto p = gir::lower_expr_to_predicate(
        gir::string_op(gir::StringOpKind::Equal, std::move(s), {"x"}),
        b.schema);
    EXPECT_EQ(p.type(), gamax::select::PredicateNodeType::STRING_EQUAL);
    EXPECT_EQ(p.string_cid(), 1);  // ordinal 1 in this schema
}

TEST(IrLowerExpr, ContainsMultipleLowersToStringContainsMultiple) {
    // `StringOpKind::ContainsMultiple` is the IR-level name for LIKE-
    // style `%p1%p2%...%` patterns; it must lower to the
    // `STRING_CONTAINS_MULTIPLE` predicate kind so the phys layer
    // reaches `gamax::contains_multiple` (which enforces sequenced
    // non-overlapping match, unlike a naive AND of Contains leaves).
    // Q16's supplier-comment filter relies on this mapping.
    auto b = make_schema({t(type_id::INT32), t(type_id::STRING)});
    auto s = gir::col_ref(b.ids[1], t(type_id::STRING), false);
    auto p = gir::lower_expr_to_predicate(
        gir::string_op(gir::StringOpKind::ContainsMultiple, std::move(s),
                       {"Customer", "Complaints"}),
        b.schema);
    EXPECT_EQ(p.type(),
              gamax::select::PredicateNodeType::STRING_CONTAINS_MULTIPLE);
    EXPECT_EQ(p.string_cid(), 1);
    ASSERT_EQ(p.string_values().size(), 2u);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
