#include <sstream>
#include <string>

#include "expr/expr_fast.hpp"
#include "memory/scalar.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax::expression {
namespace {
std::string scalar_to_string(const gamax::scalar& s, gamax::type_id id) {
    std::ostringstream oss;
    switch (id) {
        case gamax::type_id::BOOL8: oss << (s.get<bool>() ? "true" : "false"); break;
        case gamax::type_id::INT8: oss << static_cast<int>(s.get<int8_t>()); break;
        case gamax::type_id::INT16: oss << s.get<int16_t>(); break;
        case gamax::type_id::INT32: oss << s.get<int32_t>(); break;
        case gamax::type_id::INT64: oss << s.get<int64_t>(); break;
        case gamax::type_id::UINT64: oss << static_cast<unsigned long long>(s.get<int64_t>()); break;
        case gamax::type_id::FLOAT32: oss << s.get<float>(); break;
        case gamax::type_id::FLOAT64: oss << s.get<double>(); break;
        case gamax::type_id::STRING: oss << "<string>"; break;
        case gamax::type_id::EMPTY: oss << "<empty>"; break;
        case gamax::type_id::NUM_TYPE_IDS: oss << "<num_type_ids>"; break;
    }
    return oss.str();
}

std::string expr_label(const Expr& e) {
    switch (e.id) {
        case ExprId::Constant:
            return "Constant(" + scalar_to_string(e.data.c.value, e.type.id()) + ")";
        case ExprId::Column:
            return "Column(cid=" + std::to_string(e.data.col.cid) + ")";
        case ExprId::Negate: return "Negate";
        case ExprId::LeftShift: return "LeftShift(bits=" + std::to_string(e.data.lshift.bits) + ")";
        case ExprId::RightShift: return "RightShift(bits=" + std::to_string(e.data.rshift.bits) + ")";
        case ExprId::Add: return "Add";
        case ExprId::Substract: return "Subtract";
        case ExprId::Multiply: return "Multiply";
        case ExprId::Divide: return "Divide";
        case ExprId::Sqrt: return "Sqrt";
        case ExprId::Log: return "Log";
        case ExprId::Exp: return "Exp";
        case ExprId::GreaterThan: return "GreaterThan";
        case ExprId::LessThan: return "LessThan";
        case ExprId::GreaterEqual: return "GreaterEqual";
        case ExprId::LessEqual: return "LessEqual";
        case ExprId::Equal: return "Equal";
        case ExprId::And: return "And";
        case ExprId::Or: return "Or";
        case ExprId::Select: return "Select";
    }
    return "Unknown";
}

void emit_tree(const ExprTree& tree, int depth, int indent, std::ostringstream& oss) {
    const auto& e = tree.root();
    oss << std::string(static_cast<std::size_t>(depth * indent), ' ')
        << expr_label(e) << " [type=" << gamax::to_string(e.type) << "]\n";
    for (const auto& child : tree.children()) {
        if (child) {
            emit_tree(*child, depth + 1, indent, oss);
        }
    }
}
} // namespace

std::string expr_tree_to_string(const ExprTree& tree, int indent_spaces) {
    std::ostringstream oss;
    emit_tree(tree, 0, indent_spaces, oss);
    return oss.str();
}
} // namespace gamax::expression
