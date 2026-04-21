#include <gtest/gtest.h>

#include "ir/expr.hpp"

namespace gir = gamax::ir;
using gamax::data_type;
using gamax::type_id;

namespace {
data_type t(type_id id) { return data_type{id}; }
}

TEST(IrExpr, ColRefCarriesTypeAndNullable) {
    gir::IdGenerator gen;
    auto id = gen.next();
    auto e = gir::col_ref(id, t(type_id::INT32), true);
    EXPECT_EQ(e.kind, gir::ExprKind::ColRef);
    EXPECT_EQ(e.type, t(type_id::INT32));
    EXPECT_TRUE(e.nullable);
    EXPECT_EQ(e.col_ref, id);
}

TEST(IrExpr, ColRefRejectsInvalidId) {
    gir::ColumnId unset;
    EXPECT_THROW(gir::col_ref(unset, t(type_id::INT32)), std::runtime_error);
}

TEST(IrExpr, LiteralsHaveExpectedTypes) {
    EXPECT_EQ(gir::lit(int64_t{7}).type, t(type_id::INT64));
    EXPECT_EQ(gir::lit(7).type,          t(type_id::INT64));
    EXPECT_EQ(gir::lit(3.14).type,       t(type_id::FLOAT64));
    EXPECT_EQ(gir::lit(true).type,       t(type_id::BOOL8));
    EXPECT_EQ(gir::lit("hi").type,       t(type_id::STRING));
    EXPECT_FALSE(gir::lit(7).nullable);
}

TEST(IrExpr, NumericPromotion) {
    EXPECT_EQ(gir::promote_numeric(t(type_id::INT32), t(type_id::INT64)),
              t(type_id::INT64));
    EXPECT_EQ(gir::promote_numeric(t(type_id::INT64), t(type_id::FLOAT64)),
              t(type_id::FLOAT64));
    EXPECT_EQ(gir::promote_numeric(t(type_id::FLOAT32), t(type_id::FLOAT64)),
              t(type_id::FLOAT64));
    EXPECT_THROW(gir::promote_numeric(t(type_id::STRING), t(type_id::INT32)),
                 std::runtime_error);
}

TEST(IrExpr, ArithTypeInference) {
    gir::IdGenerator gen;
    auto a = gir::col_ref(gen.next(), t(type_id::INT32));
    auto b = gir::col_ref(gen.next(), t(type_id::FLOAT64));
    auto e = a + b;
    EXPECT_EQ(e.kind, gir::ExprKind::BinArith);
    EXPECT_EQ(e.arith_op, gir::ArithOp::Add);
    EXPECT_EQ(e.type, t(type_id::FLOAT64));
    EXPECT_EQ(e.children.size(), 2u);
}

TEST(IrExpr, ArithRejectsNonNumeric) {
    gir::IdGenerator gen;
    auto s = gir::col_ref(gen.next(), t(type_id::STRING));
    EXPECT_THROW(s + gir::lit(int64_t{1}), std::runtime_error);
}

TEST(IrExpr, CompareProducesBool) {
    gir::IdGenerator gen;
    auto a = gir::col_ref(gen.next(), t(type_id::INT64));
    auto e = a < gir::lit(int64_t{10});
    EXPECT_EQ(e.kind, gir::ExprKind::BinCompare);
    EXPECT_EQ(e.compare_op, gir::CompareOp::Lt);
    EXPECT_EQ(e.type, t(type_id::BOOL8));
}

TEST(IrExpr, CompareRejectsMixedStringNumeric) {
    gir::IdGenerator gen;
    auto s = gir::col_ref(gen.next(), t(type_id::STRING));
    auto i = gir::col_ref(gen.next(), t(type_id::INT32));
    EXPECT_THROW(s < i, std::runtime_error);
}

TEST(IrExpr, LogicalAndOrNot) {
    gir::IdGenerator gen;
    auto a = gir::col_ref(gen.next(), t(type_id::INT64));
    auto b = gir::col_ref(gen.next(), t(type_id::INT64));
    auto p = (a >= gir::lit(int64_t{1})) && (b < gir::lit(int64_t{10}));
    EXPECT_EQ(p.kind, gir::ExprKind::BinLogical);
    EXPECT_EQ(p.type, t(type_id::BOOL8));
    auto q = !p;
    EXPECT_EQ(q.kind, gir::ExprKind::Unary);
    EXPECT_EQ(q.unary_op, gir::UnaryOp::Not);
}

TEST(IrExpr, LogicalRejectsNonBool) {
    gir::IdGenerator gen;
    auto a = gir::col_ref(gen.next(), t(type_id::INT32));
    auto b = gir::col_ref(gen.next(), t(type_id::INT32));
    EXPECT_THROW(a && b, std::runtime_error);
    EXPECT_THROW(!a, std::runtime_error);
}

TEST(IrExpr, CaseWhenUnifiesBranches) {
    gir::IdGenerator gen;
    auto cond = gir::col_ref(gen.next(), t(type_id::BOOL8));
    auto then_ = gir::col_ref(gen.next(), t(type_id::FLOAT64));
    auto else_ = gir::lit(0.0);
    auto e = gir::case_when(std::move(cond), std::move(then_), std::move(else_));
    EXPECT_EQ(e.kind, gir::ExprKind::Case);
    EXPECT_EQ(e.type, t(type_id::FLOAT64));
    EXPECT_EQ(e.children.size(), 3u);
}

TEST(IrExpr, CaseWhenPromotesNumericBranches) {
    gir::IdGenerator gen;
    auto cond = gir::col_ref(gen.next(), t(type_id::BOOL8));
    auto e = gir::case_when(std::move(cond),
                            gir::lit(int64_t{1}),
                            gir::lit(2.0));
    EXPECT_EQ(e.type, t(type_id::FLOAT64));
}

TEST(IrExpr, CaseWhenRejectsIncompatibleBranches) {
    gir::IdGenerator gen;
    auto cond = gir::col_ref(gen.next(), t(type_id::BOOL8));
    EXPECT_THROW(
        gir::case_when(std::move(cond), gir::lit(int64_t{1}), gir::lit("s")),
        std::runtime_error);
}

TEST(IrExpr, StringOpSingleValue) {
    gir::IdGenerator gen;
    auto c = gir::col_ref(gen.next(), t(type_id::STRING));
    auto e = gir::string_op(gir::StringOpKind::Startswith, std::move(c), {"PROMO"});
    EXPECT_EQ(e.kind, gir::ExprKind::StringOp);
    EXPECT_EQ(e.string_op, gir::StringOpKind::Startswith);
    EXPECT_EQ(e.type, t(type_id::BOOL8));
    EXPECT_EQ(e.string_values.size(), 1u);
}

TEST(IrExpr, StringOpInRequiresValues) {
    gir::IdGenerator gen;
    auto c = gir::col_ref(gen.next(), t(type_id::STRING));
    EXPECT_THROW(gir::string_op(gir::StringOpKind::In, std::move(c), {}),
                 std::runtime_error);
}

TEST(IrExpr, StringOpRejectsNonStringColumn) {
    gir::IdGenerator gen;
    auto c = gir::col_ref(gen.next(), t(type_id::INT32));
    EXPECT_THROW(gir::string_op(gir::StringOpKind::Equal, std::move(c), {"x"}),
                 std::runtime_error);
}

TEST(IrExpr, NullablePropagation) {
    gir::IdGenerator gen;
    auto a = gir::col_ref(gen.next(), t(type_id::INT64), /*nullable=*/true);
    auto b = gir::col_ref(gen.next(), t(type_id::INT64), /*nullable=*/false);
    EXPECT_TRUE((a + b).nullable);
    EXPECT_TRUE((a == b).nullable);
    auto both_bool = (a == b) && (b == b);
    EXPECT_TRUE(both_bool.nullable);
}

// Rebuilds Q6's filter as a shape-check that the IR can represent a
// real query's predicate cleanly.
TEST(IrExpr, Q6FilterShape) {
    gir::IdGenerator gen;
    auto shipdate   = gir::col_ref(gen.next(), t(type_id::INT32));
    auto price      = gir::col_ref(gen.next(), t(type_id::FLOAT64));
    auto discount   = gir::col_ref(gen.next(), t(type_id::FLOAT64));
    auto quantity   = gir::col_ref(gen.next(), t(type_id::INT32));

    auto filter = (shipdate  >= gir::lit(8766))
               && (shipdate  <  gir::lit(9131))
               && (discount  >= gir::lit(0.05))
               && (discount  <= gir::lit(0.07))
               && (quantity  <  gir::lit(24));

    EXPECT_EQ(filter.type, t(type_id::BOOL8));
    EXPECT_EQ(filter.kind, gir::ExprKind::BinLogical);

    auto project = price * discount;
    EXPECT_EQ(project.type, t(type_id::FLOAT64));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
