#include "ir/expr.hpp"

#include <stdexcept>
#include <string>

namespace gamax::ir {

namespace {

// Numeric-type rank for widening promotion.
// Higher rank == wider type. Non-numeric types return -1.
int numeric_rank(gamax::type_id t) {
    switch (t) {
        case gamax::type_id::INT8:    return 1;
        case gamax::type_id::INT16:   return 2;
        case gamax::type_id::INT32:   return 3;
        case gamax::type_id::INT64:   return 4;
        case gamax::type_id::UINT8:   return 1;
        case gamax::type_id::UINT16:  return 2;
        case gamax::type_id::UINT32:  return 3;
        case gamax::type_id::UINT64:  return 4;
        case gamax::type_id::FLOAT32: return 5;
        case gamax::type_id::FLOAT64: return 6;
        default: return -1;
    }
}

const char* type_name(gamax::type_id t) {
    switch (t) {
        case gamax::type_id::EMPTY:   return "EMPTY";
        case gamax::type_id::INT8:    return "INT8";
        case gamax::type_id::INT16:   return "INT16";
        case gamax::type_id::INT32:   return "INT32";
        case gamax::type_id::INT64:   return "INT64";
        case gamax::type_id::BOOL8:   return "BOOL8";
        case gamax::type_id::UINT8:   return "UINT8";
        case gamax::type_id::UINT16:  return "UINT16";
        case gamax::type_id::UINT32:  return "UINT32";
        case gamax::type_id::UINT64:  return "UINT64";
        case gamax::type_id::FLOAT32: return "FLOAT32";
        case gamax::type_id::FLOAT64: return "FLOAT64";
        case gamax::type_id::STRING:  return "STRING";
        default: return "<unknown>";
    }
}

[[noreturn]] void type_error(const std::string& what) {
    throw std::runtime_error("ir::Expr: " + what);
}

} // namespace

bool is_numeric(gamax::type_id t) { return numeric_rank(t) >= 0; }

bool is_integer(gamax::type_id t) {
    switch (t) {
        case gamax::type_id::INT8:   case gamax::type_id::INT16:
        case gamax::type_id::INT32:  case gamax::type_id::INT64:
        case gamax::type_id::UINT8:  case gamax::type_id::UINT16:
        case gamax::type_id::UINT32: case gamax::type_id::UINT64:
            return true;
        default:
            return false;
    }
}

gamax::data_type promote_numeric(gamax::data_type a, gamax::data_type b) {
    int ra = numeric_rank(a.id());
    int rb = numeric_rank(b.id());
    if (ra < 0 || rb < 0) {
        type_error(std::string("promote_numeric: non-numeric operand (")
                   + type_name(a.id()) + ", " + type_name(b.id()) + ")");
    }
    return ra >= rb ? a : b;
}

// ── Factories ─────────────────────────────────────────────────────

Expr col_ref(ColumnId id, gamax::data_type type, bool nullable) {
    if (!id.is_valid()) {
        type_error("col_ref: invalid ColumnId (is the column wired up?)");
    }
    Expr e;
    e.kind = ExprKind::ColRef;
    e.type = type;
    e.nullable = nullable;
    e.col_ref = id;
    return e;
}

namespace {
Expr make_literal(LiteralValue v, gamax::data_type type) {
    Expr e;
    e.kind = ExprKind::Literal;
    e.type = type;
    e.nullable = false;  // a literal is never NULL in our IR — a
                         // SQL NULL literal would need its own Kind.
    e.literal = std::move(v);
    return e;
}
} // namespace

Expr lit(int64_t v) {
    return make_literal(v, gamax::data_type{gamax::type_id::INT64});
}
Expr lit(int v) { return lit(static_cast<int64_t>(v)); }
Expr lit(double v) {
    return make_literal(v, gamax::data_type{gamax::type_id::FLOAT64});
}
Expr lit(bool v) {
    return make_literal(v, gamax::data_type{gamax::type_id::BOOL8});
}
Expr lit(std::string v) {
    return make_literal(std::move(v), gamax::data_type{gamax::type_id::STRING});
}
Expr lit(const char* v) { return lit(std::string(v)); }

Expr bin_arith(ArithOp op, Expr l, Expr r) {
    auto lt = l.type.id();
    auto rt = r.type.id();
    if (!is_numeric(lt) || !is_numeric(rt)) {
        type_error(std::string("bin_arith: non-numeric operand (")
                   + type_name(lt) + ", " + type_name(rt) + ")");
    }
    Expr e;
    e.kind = ExprKind::BinArith;
    e.arith_op = op;
    e.type = promote_numeric(l.type, r.type);
    e.nullable = l.nullable || r.nullable;
    e.children.push_back(std::move(l));
    e.children.push_back(std::move(r));
    return e;
}

Expr bin_compare(CompareOp op, Expr l, Expr r) {
    auto lt = l.type.id();
    auto rt = r.type.id();
    const bool both_numeric = is_numeric(lt) && is_numeric(rt);
    const bool both_string  = (lt == gamax::type_id::STRING)
                           && (rt == gamax::type_id::STRING);
    const bool both_bool    = (lt == gamax::type_id::BOOL8)
                           && (rt == gamax::type_id::BOOL8);
    if (!both_numeric && !both_string && !both_bool) {
        type_error(std::string("bin_compare: incomparable operands (")
                   + type_name(lt) + ", " + type_name(rt) + ")");
    }
    Expr e;
    e.kind = ExprKind::BinCompare;
    e.compare_op = op;
    e.type = gamax::data_type{gamax::type_id::BOOL8};
    e.nullable = l.nullable || r.nullable;
    e.children.push_back(std::move(l));
    e.children.push_back(std::move(r));
    return e;
}

Expr bin_logical(LogicalOp op, Expr l, Expr r) {
    if (l.type.id() != gamax::type_id::BOOL8
        || r.type.id() != gamax::type_id::BOOL8) {
        type_error(std::string("bin_logical: requires BOOL8 operands, got (")
                   + type_name(l.type.id()) + ", "
                   + type_name(r.type.id()) + ")");
    }
    Expr e;
    e.kind = ExprKind::BinLogical;
    e.logical_op = op;
    e.type = gamax::data_type{gamax::type_id::BOOL8};
    e.nullable = l.nullable || r.nullable;
    e.children.push_back(std::move(l));
    e.children.push_back(std::move(r));
    return e;
}

Expr unary_not(Expr child) {
    if (child.type.id() != gamax::type_id::BOOL8) {
        type_error(std::string("unary_not: requires BOOL8 operand, got ")
                   + type_name(child.type.id()));
    }
    Expr e;
    e.kind = ExprKind::Unary;
    e.unary_op = UnaryOp::Not;
    e.type = gamax::data_type{gamax::type_id::BOOL8};
    e.nullable = child.nullable;
    e.children.push_back(std::move(child));
    return e;
}

Expr case_when(Expr cond, Expr then_, Expr else_) {
    if (cond.type.id() != gamax::type_id::BOOL8) {
        type_error(std::string("case_when: condition must be BOOL8, got ")
                   + type_name(cond.type.id()));
    }
    gamax::data_type result_type;
    if (then_.type == else_.type) {
        result_type = then_.type;
    } else if (is_numeric(then_.type.id()) && is_numeric(else_.type.id())) {
        result_type = promote_numeric(then_.type, else_.type);
    } else {
        type_error(std::string("case_when: then/else types do not unify (")
                   + type_name(then_.type.id()) + ", "
                   + type_name(else_.type.id()) + ")");
    }
    Expr e;
    e.kind = ExprKind::Case;
    e.type = result_type;
    e.nullable = cond.nullable || then_.nullable || else_.nullable;
    e.children.push_back(std::move(cond));
    e.children.push_back(std::move(then_));
    e.children.push_back(std::move(else_));
    return e;
}

Expr string_op(StringOpKind op, Expr col, std::vector<std::string> values) {
    if (col.type.id() != gamax::type_id::STRING) {
        type_error(std::string("string_op: column operand must be STRING, got ")
                   + type_name(col.type.id()));
    }
    const bool multi = (op == StringOpKind::ContainsAny)
                    || (op == StringOpKind::In);
    if (!multi && values.size() != 1) {
        type_error("string_op: single-value operator requires exactly one needle");
    }
    if (multi && values.empty()) {
        type_error("string_op: multi-value operator requires at least one needle");
    }
    Expr e;
    e.kind = ExprKind::StringOp;
    e.string_op = op;
    e.string_values = std::move(values);
    e.type = gamax::data_type{gamax::type_id::BOOL8};
    e.nullable = col.nullable;
    e.children.push_back(std::move(col));
    return e;
}

// ── Operator overloads ────────────────────────────────────────────

Expr operator+(Expr l, Expr r) { return bin_arith(ArithOp::Add, std::move(l), std::move(r)); }
Expr operator-(Expr l, Expr r) { return bin_arith(ArithOp::Sub, std::move(l), std::move(r)); }
Expr operator*(Expr l, Expr r) { return bin_arith(ArithOp::Mul, std::move(l), std::move(r)); }
Expr operator/(Expr l, Expr r) { return bin_arith(ArithOp::Div, std::move(l), std::move(r)); }

Expr operator==(Expr l, Expr r) { return bin_compare(CompareOp::Eq, std::move(l), std::move(r)); }
Expr operator!=(Expr l, Expr r) { return bin_compare(CompareOp::Ne, std::move(l), std::move(r)); }
Expr operator< (Expr l, Expr r) { return bin_compare(CompareOp::Lt, std::move(l), std::move(r)); }
Expr operator<=(Expr l, Expr r) { return bin_compare(CompareOp::Le, std::move(l), std::move(r)); }
Expr operator> (Expr l, Expr r) { return bin_compare(CompareOp::Gt, std::move(l), std::move(r)); }
Expr operator>=(Expr l, Expr r) { return bin_compare(CompareOp::Ge, std::move(l), std::move(r)); }

Expr operator&&(Expr l, Expr r) { return bin_logical(LogicalOp::And, std::move(l), std::move(r)); }
Expr operator||(Expr l, Expr r) { return bin_logical(LogicalOp::Or,  std::move(l), std::move(r)); }
Expr operator!(Expr e)          { return unary_not(std::move(e)); }

} // namespace gamax::ir
