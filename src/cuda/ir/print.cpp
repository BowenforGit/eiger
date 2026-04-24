#include "ir/print.hpp"

#include <sstream>
#include <variant>

#include "types/data_type.hpp"

namespace gamax::ir {

namespace {

// ── Type / op / kind name tables ───────────────────────────────────

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
        default: return "?";
    }
}

const char* arith_name(ArithOp op) {
    switch (op) {
        case ArithOp::Add: return "+";
        case ArithOp::Sub: return "-";
        case ArithOp::Mul: return "*";
        case ArithOp::Div: return "/";
    }
    return "?";
}
const char* compare_name(CompareOp op) {
    switch (op) {
        case CompareOp::Eq: return "==";
        case CompareOp::Ne: return "!=";
        case CompareOp::Lt: return "<";
        case CompareOp::Le: return "<=";
        case CompareOp::Gt: return ">";
        case CompareOp::Ge: return ">=";
    }
    return "?";
}
const char* logical_name(LogicalOp op) {
    switch (op) {
        case LogicalOp::And: return "&&";
        case LogicalOp::Or:  return "||";
    }
    return "?";
}
const char* string_op_name(StringOpKind op) {
    switch (op) {
        case StringOpKind::Equal:        return "str_eq";
        case StringOpKind::Startswith:   return "startswith";
        case StringOpKind::Endswith:     return "endswith";
        case StringOpKind::Contains:     return "contains";
        case StringOpKind::ContainsMultiple: return "contains_multiple";
        case StringOpKind::In:           return "str_in";
    }
    return "?";
}
const char* agg_name(AggKind k) {
    switch (k) {
        case AggKind::Sum:           return "sum";
        case AggKind::Avg:           return "avg";
        case AggKind::Min:           return "min";
        case AggKind::Max:           return "max";
        case AggKind::Count:         return "count";
        case AggKind::CountAll:      return "count_all";
        case AggKind::CountDistinct: return "count_distinct";
    }
    return "?";
}
const char* join_name(JoinKind k) {
    switch (k) {
        case JoinKind::Inner:     return "InnerJoin";
        case JoinKind::LeftOuter: return "LeftOuterJoin";
        case JoinKind::LeftSemi:  return "LeftSemiJoin";
        case JoinKind::LeftAnti:  return "LeftAntiJoin";
        case JoinKind::Cross:     return "CrossJoin";
    }
    return "?";
}
const char* order_name(gamax::sort_order o) {
    return o == gamax::sort_order::ASCENDING ? "ASC" : "DESC";
}

// ── Expression formatting ─────────────────────────────────────────

void write_literal(std::ostream& os, const LiteralValue& v) {
    std::visit([&os](auto&& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, int64_t>) {
            os << x;
        } else if constexpr (std::is_same_v<T, double>) {
            os << x;
        } else if constexpr (std::is_same_v<T, bool>) {
            os << (x ? "true" : "false");
        } else if constexpr (std::is_same_v<T, std::string>) {
            os << '"' << x << '"';
        }
    }, v);
}

void write_col_ref(
    std::ostream& os,
    ColumnId id,
    const std::unordered_map<ColumnId, std::string>& names)
{
    auto it = names.find(id);
    if (it != names.end() && !it->second.empty()) {
        os << it->second;
    } else {
        os << "#" << id.value;
    }
}

// Forward decl: expr printer is mutually recursive.
void write_expr(
    std::ostream& os,
    const Expr& e,
    const std::unordered_map<ColumnId, std::string>& names);

void write_string_list(std::ostream& os, const std::vector<std::string>& xs) {
    os << "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) os << ", ";
        os << '"' << xs[i] << '"';
    }
    os << "]";
}

void write_expr(
    std::ostream& os,
    const Expr& e,
    const std::unordered_map<ColumnId, std::string>& names)
{
    switch (e.kind) {
        case ExprKind::ColRef:
            write_col_ref(os, e.col_ref, names);
            return;
        case ExprKind::Literal:
            write_literal(os, e.literal);
            return;
        case ExprKind::BinArith: {
            os << "(";
            write_expr(os, e.children[0], names);
            os << " " << arith_name(e.arith_op) << " ";
            write_expr(os, e.children[1], names);
            os << ")";
            return;
        }
        case ExprKind::BinCompare: {
            os << "(";
            write_expr(os, e.children[0], names);
            os << " " << compare_name(e.compare_op) << " ";
            write_expr(os, e.children[1], names);
            os << ")";
            return;
        }
        case ExprKind::BinLogical: {
            os << "(";
            write_expr(os, e.children[0], names);
            os << " " << logical_name(e.logical_op) << " ";
            write_expr(os, e.children[1], names);
            os << ")";
            return;
        }
        case ExprKind::Unary: {
            os << "!";
            write_expr(os, e.children[0], names);
            return;
        }
        case ExprKind::StringOp: {
            os << string_op_name(e.string_op) << "(";
            write_expr(os, e.children[0], names);
            os << ", ";
            if (e.string_values.size() == 1) {
                os << '"' << e.string_values[0] << '"';
            } else {
                write_string_list(os, e.string_values);
            }
            os << ")";
            return;
        }
        case ExprKind::Case: {
            os << "case(";
            write_expr(os, e.children[0], names);
            os << ", ";
            write_expr(os, e.children[1], names);
            os << ", ";
            write_expr(os, e.children[2], names);
            os << ")";
            return;
        }
        case ExprKind::FunctionCall: {
            os << e.function_name << "(";
            for (size_t i = 0; i < e.children.size(); ++i) {
                if (i) os << ", ";
                write_expr(os, e.children[i], names);
            }
            os << ")";
            return;
        }
    }
    os << "<?expr?>";
}

// ── Plan formatting ───────────────────────────────────────────────

void collect_names(
    const PlanNode& node,
    std::unordered_map<ColumnId, std::string>& names)
{
    for (const auto& f : node.output_schema.fields()) {
        if (!f.name.empty()) names[f.id] = f.name;
    }
    for (const auto& in : node.inputs) {
        if (in) collect_names(*in, names);
    }
}

// After `collect_names` has populated the id -> name map, post-process
// it so that names shared by more than one ColumnId get a `#<id>`
// suffix. Without this, a self-join (or any two scans of the same
// table) renders both sides' columns with identical labels, which is
// misleading — the emitter sees them as distinct ids. Unambiguous
// names pass through unchanged so common plans read cleanly.
void disambiguate_names(std::unordered_map<ColumnId, std::string>& names) {
    std::unordered_map<std::string, int> counts;
    for (const auto& [_, n] : names) {
        if (!n.empty()) counts[n]++;
    }
    for (auto& [id, n] : names) {
        if (!n.empty() && counts[n] > 1) {
            n += "#" + std::to_string(id.value);
        }
    }
}

void write_indent(std::ostream& os, int depth) {
    for (int i = 0; i < depth; ++i) os << "  ";
}

void write_cols_brief(
    std::ostream& os,
    const std::vector<ColumnId>& ids,
    const std::unordered_map<ColumnId, std::string>& names)
{
    os << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) os << ", ";
        write_col_ref(os, ids[i], names);
    }
    os << "]";
}

void write_node_line(
    std::ostream& os,
    const PlanNode& n,
    const std::unordered_map<ColumnId, std::string>& names)
{
    switch (n.kind) {
        case NodeKind::Scan: {
            os << "Scan(" << n.scan_table << " [";
            for (size_t i = 0; i < n.scan_columns.size(); ++i) {
                if (i) os << ", ";
                os << n.scan_columns[i];
            }
            os << "])";
            return;
        }
        case NodeKind::Filter:
            os << "Filter(";
            write_expr(os, n.filter_predicate, names);
            os << ")";
            return;
        case NodeKind::Project: {
            os << "Project(";
            for (size_t i = 0; i < n.project_items.size(); ++i) {
                if (i) os << ", ";
                os << n.project_items[i].name << " = ";
                write_expr(os, n.project_items[i].expr, names);
            }
            os << ")";
            return;
        }
        case NodeKind::Aggregate: {
            os << "Aggregate(";
            if (!n.agg_keys.empty()) {
                os << "keys=";
                write_cols_brief(os, n.agg_keys, names);
                if (!n.agg_items.empty()) os << "; ";
            }
            os << "aggs=[";
            for (size_t i = 0; i < n.agg_items.size(); ++i) {
                if (i) os << ", ";
                const auto& a = n.agg_items[i];
                os << agg_name(a.agg) << "(";
                if (a.agg == AggKind::CountAll) {
                    os << "*";
                } else {
                    write_expr(os, a.body, names);
                }
                os << ") AS " << a.name;
            }
            os << "])";
            return;
        }
        case NodeKind::Sort: {
            os << "Sort[";
            for (size_t i = 0; i < n.sort_keys.size(); ++i) {
                if (i) os << ", ";
                write_col_ref(os, n.sort_keys[i].id, names);
                os << " " << order_name(n.sort_keys[i].order);
            }
            os << "]";
            return;
        }
        case NodeKind::Limit:
            os << "Limit(n=" << n.limit_n;
            if (n.limit_offset != 0) os << ", offset=" << n.limit_offset;
            os << ")";
            return;
        case NodeKind::Join: {
            os << join_name(n.join_kind);
            if (n.join_condition) {
                os << "(on=";
                write_expr(os, *n.join_condition, names);
                os << ")";
            }
            return;
        }
        case NodeKind::Distinct:
            os << "Distinct(";
            if (n.distinct_keys.empty()) {
                os << "all";
            } else {
                os << "keys=";
                write_cols_brief(os, n.distinct_keys, names);
            }
            os << ")";
            return;
        case NodeKind::Concat:
            os << "Concat(#inputs=" << n.inputs.size() << ")";
            return;
    }
    os << "<?node?>";
}

void write_plan_rec(
    std::ostream& os,
    const PlanNode& n,
    int depth,
    const std::unordered_map<ColumnId, std::string>& names)
{
    write_indent(os, depth);
    write_node_line(os, n, names);
    os << "\n";
    for (const auto& in : n.inputs) {
        if (in) write_plan_rec(os, *in, depth + 1, names);
    }
}

} // namespace

std::string format_expr(const Expr& e) {
    std::unordered_map<ColumnId, std::string> empty;
    std::ostringstream os;
    write_expr(os, e, empty);
    return os.str();
}

std::string format_expr(
    const Expr& e,
    const std::unordered_map<ColumnId, std::string>& names)
{
    std::ostringstream os;
    write_expr(os, e, names);
    return os.str();
}

std::string format_schema(const Schema& s) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) os << ", ";
        const auto& f = s[i];
        if (!f.name.empty()) os << f.name; else os << "#" << f.id.value;
        os << ":" << type_name(f.type.id());
    }
    os << "]";
    return os.str();
}

std::string format_plan(const PlanNode& root) {
    std::ostringstream os;
    print_plan(os, root);
    return os.str();
}

void print_plan(std::ostream& os, const PlanNode& root) {
    std::unordered_map<ColumnId, std::string> names;
    collect_names(root, names);
    disambiguate_names(names);
    write_plan_rec(os, root, 0, names);
}

} // namespace gamax::ir
