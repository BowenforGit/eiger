#pragma once

#include <string>
#include <variant>
#include <vector>

#include "ir/column_id.hpp"
#include "types/data_type.hpp"

namespace gamax::ir {

// Tag describing which Expr variant a node represents. Only the payload
// field(s) named in the per-kind comments on Expr are meaningful; the
// rest are left in their default-constructed state.
enum class ExprKind {
    ColRef,
    Literal,
    BinArith,
    BinCompare,
    BinLogical,
    Unary,
    StringOp,
    Case,
    FunctionCall,   // name-tagged scalar function: year/extract/substring/...
    // Subquery and AggRef are reserved for later phases; their node
    // kinds will be added when HAVING / ORDER-BY-over-aggregates /
    // correlated subqueries become in scope.
};

enum class ArithOp   { Add, Sub, Mul, Div };
enum class CompareOp { Eq, Ne, Lt, Le, Gt, Ge };
enum class LogicalOp { And, Or };
enum class UnaryOp   { Not };
enum class StringOpKind {
    Equal, Startswith, Endswith, Contains, ContainsMultiple, In,
};

// Literal value carried by an Expr of kind Literal.
//
// Integer literals are stored as int64 and narrowed at emit time;
// floating-point literals are stored as double and narrowed similarly.
// String literals are plain std::string at the IR level and only become
// a gamax::string_scalar when a physical string operator is invoked.
using LiteralValue = std::variant<int64_t, double, bool, std::string>;

// A typed scalar expression over plan-node output columns.
//
// Children are held by value. This makes the tree trivially cloneable
// (important for rule-based rewrites) at the cost of O(tree size) per
// copy; expressions in practice are shallow so this is cheap. Every
// Expr has a concrete `type` and `nullable` flag inferred bottom-up at
// construction time via the factory functions in this header — the
// default-constructed Expr is not meaningful on its own.
struct Expr {
    ExprKind kind {ExprKind::Literal};
    gamax::data_type type;
    bool nullable {false};
    std::vector<Expr> children;

    // Per-kind payload. Exactly one field is meaningful per kind; the
    // others are left default. Kept as named fields rather than a
    // std::variant so that the shape mirrors the existing Predicate /
    // aggregation_request conventions in this codebase.
    ColumnId      col_ref;                            // ColRef
    LiteralValue  literal;                            // Literal
    ArithOp       arith_op   {ArithOp::Add};          // BinArith
    CompareOp     compare_op {CompareOp::Eq};         // BinCompare
    LogicalOp     logical_op {LogicalOp::And};        // BinLogical
    UnaryOp       unary_op   {UnaryOp::Not};          // Unary
    StringOpKind  string_op  {StringOpKind::Equal};   // StringOp
    std::vector<std::string> string_values;           // StringOp (In/ContainsMultiple)
    std::string   function_name;                      // FunctionCall
};

// ── Factories (each runs type inference and validation) ────────────

// Column reference. Callers pass the Field's type and nullable flag
// from the input schema; during plan construction the Builder does this
// lookup automatically.
Expr col_ref(ColumnId id, gamax::data_type type, bool nullable = false);

// Numeric / boolean / string literals. `int` overload exists because
// C++ picks int over int64_t for integer constants.
Expr lit(int64_t v);
Expr lit(int v);
Expr lit(double v);
Expr lit(bool v);
Expr lit(std::string v);
Expr lit(const char* v);

// Arithmetic. Operands must be numeric; result type is the wider of
// the two (FLOAT64 > FLOAT32 > INT64 > ... > INT8).
Expr bin_arith(ArithOp op, Expr l, Expr r);

// Comparison. Operands must be mutually comparable (numeric/numeric or
// string/string or bool/bool). Result type is BOOL8.
Expr bin_compare(CompareOp op, Expr l, Expr r);

// Boolean combinator. Both operands must be BOOL8. Result is BOOL8.
Expr bin_logical(LogicalOp op, Expr l, Expr r);

// Boolean negation. Child must be BOOL8.
Expr unary_not(Expr child);

// SQL CASE-WHEN / ternary. `cond` must be BOOL8; `then` and `else` must
// share a common type (numeric promotion is applied). Result carries
// that common type.
Expr case_when(Expr cond, Expr then_, Expr else_);

// String operation. `col` is the string column being matched; `values`
// supplies the needle(s) (one for Equal/Startswith/Endswith/Contains,
// possibly many for ContainsMultiple/In). Result is BOOL8.
Expr string_op(StringOpKind op, Expr col, std::vector<std::string> values);

// Name-tagged scalar function call. The IR tracks the `name` (e.g.
// "year", "substring", "extract_year") and the argument list; the
// declared `return_type` / `nullable` are taken on faith, since the IR
// deliberately stays generic about built-in functions (a later pass
// / emitter resolves the name to a concrete physical implementation
// or errors with "unknown function").
//
// This is the escape hatch for the long tail of scalar functions
// (date/time, math, string manipulation) without baking each into
// ExprKind.
Expr func(std::string name,
          std::vector<Expr> args,
          gamax::data_type return_type,
          bool nullable = false);

// ── Operator overloads (for fluent construction) ───────────────────

Expr operator+(Expr l, Expr r);
Expr operator-(Expr l, Expr r);
Expr operator*(Expr l, Expr r);
Expr operator/(Expr l, Expr r);
Expr operator==(Expr l, Expr r);
Expr operator!=(Expr l, Expr r);
Expr operator<(Expr l, Expr r);
Expr operator<=(Expr l, Expr r);
Expr operator>(Expr l, Expr r);
Expr operator>=(Expr l, Expr r);
Expr operator&&(Expr l, Expr r);
Expr operator||(Expr l, Expr r);
Expr operator!(Expr e);

// ── Type utilities exposed for use by plan-node payload validation ──

bool is_numeric(gamax::type_id t);
bool is_integer(gamax::type_id t);

// Widening promotion over the numeric portion of the type lattice.
// Throws if either argument is non-numeric.
gamax::data_type promote_numeric(gamax::data_type a, gamax::data_type b);

// All ColumnIds referenced by ColRef leaves of `e`, in in-order
// traversal. Used by PlanNode factories to validate that an expression
// only mentions columns in the input schema.
std::vector<ColumnId> collect_column_refs(const Expr& e);

} // namespace gamax::ir
