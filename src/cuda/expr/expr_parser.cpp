#include "expr/expr_parser.hpp"

#include <cctype>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace gamax::expression {
namespace {

constexpr gamax::data_type kBoolType{gamax::type_id::BOOL8};

enum class TokenKind {
    Identifier,
    Number,
    LParen,
    RParen,
    Question,
    Colon,
    Plus,
    Minus,
    Star,
    Slash,
    ShiftLeft,
    ShiftRight,
    Greater,
    Less,
    GreaterEqual,
    LessEqual,
    EqualEqual,
    AndAnd,
    OrOr,
    Comma,
    End
};

struct Token {
    TokenKind kind {TokenKind::End};
    std::string text;
};

bool is_identifier_start(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool is_identifier_body(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

bool is_type_name(std::string_view id) {
    return id == "int8" || id == "int16" || id == "int32" || id == "int64" ||
           id == "float32" || id == "float64" || id == "bool" || id == "bool8";
}

bool is_function_name(std::string_view id) {
    return id == "sqrt" || id == "log" || id == "exp";
}

std::string to_lower(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

gamax::data_type type_from_name(std::string_view name) {
    auto lowered = to_lower(name);
    if (lowered == "int8") return gamax::data_type(gamax::type_id::INT8);
    if (lowered == "int16") return gamax::data_type(gamax::type_id::INT16);
    if (lowered == "int32") return gamax::data_type(gamax::type_id::INT32);
    if (lowered == "int64") return gamax::data_type(gamax::type_id::INT64);
    if (lowered == "float32") return gamax::data_type(gamax::type_id::FLOAT32);
    if (lowered == "float64") return gamax::data_type(gamax::type_id::FLOAT64);
    if (lowered == "bool" || lowered == "bool8") return gamax::data_type(gamax::type_id::BOOL8);
    throw std::runtime_error("Unknown type name: " + std::string(name));
}

struct LiteralInfo {
    gamax::scalar value {};
    gamax::data_type type {};
};

LiteralInfo make_numeric_literal(const std::string& text, std::optional<gamax::data_type> cast_type) {
    const bool is_float_literal = text.find('.') != std::string::npos ||
                                  text.find('e') != std::string::npos ||
                                  text.find('E') != std::string::npos;

    if (cast_type.has_value()) {
        switch (cast_type->id()) {
            case gamax::type_id::INT8:   return {gamax::scalar(static_cast<int8_t>(std::stoll(text))), *cast_type};
            case gamax::type_id::INT16:  return {gamax::scalar(static_cast<int16_t>(std::stoll(text))), *cast_type};
            case gamax::type_id::INT32:  return {gamax::scalar(static_cast<int32_t>(std::stoll(text))), *cast_type};
            case gamax::type_id::INT64:  return {gamax::scalar(static_cast<int64_t>(std::stoll(text))), *cast_type};
            case gamax::type_id::FLOAT32: return {gamax::scalar(static_cast<float>(std::stod(text))), *cast_type};
            case gamax::type_id::FLOAT64: return {gamax::scalar(std::stod(text)), *cast_type};
            case gamax::type_id::BOOL8: {
                auto numeric = std::stoll(text);
                return {gamax::scalar(static_cast<bool>(numeric != 0)), *cast_type};
            }
            default:
                throw std::runtime_error("Unsupported cast type for numeric literal");
        }
    }

    if (is_float_literal) {
        return {gamax::scalar(std::stod(text)), gamax::data_type(gamax::type_id::FLOAT64)};
    }

    return {gamax::scalar(static_cast<int32_t>(std::stoll(text))), gamax::data_type(gamax::type_id::INT32)};
}

std::vector<Token> tokenize(std::string_view input) {
    std::vector<Token> tokens;
    std::size_t i = 0;
    const std::size_t n = input.size();

    while (i < n) {
        char c = input[i];

        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }

        // Two-character operators
        if (i + 1 < n) {
            char next = input[i + 1];
            if (c == '<' && next == '<') { tokens.push_back({TokenKind::ShiftLeft, "<<"}); i += 2; continue; }
            if (c == '>' && next == '>') { tokens.push_back({TokenKind::ShiftRight, ">>"}); i += 2; continue; }
            if (c == '>' && next == '=') { tokens.push_back({TokenKind::GreaterEqual, ">="}); i += 2; continue; }
            if (c == '<' && next == '=') { tokens.push_back({TokenKind::LessEqual, "<="}); i += 2; continue; }
            if (c == '=' && next == '=') { tokens.push_back({TokenKind::EqualEqual, "=="}); i += 2; continue; }
            if (c == '&' && next == '&') { tokens.push_back({TokenKind::AndAnd, "&&"}); i += 2; continue; }
            if (c == '|' && next == '|') { tokens.push_back({TokenKind::OrOr, "||"}); i += 2; continue; }
        }

        switch (c) {
            case '(': tokens.push_back({TokenKind::LParen, "("}); ++i; continue;
            case ')': tokens.push_back({TokenKind::RParen, ")"}); ++i; continue;
            case '?': tokens.push_back({TokenKind::Question, "?"}); ++i; continue;
            case ':': tokens.push_back({TokenKind::Colon, ":"}); ++i; continue;
            case '+': tokens.push_back({TokenKind::Plus, "+"}); ++i; continue;
            case '-': tokens.push_back({TokenKind::Minus, "-"}); ++i; continue;
            case '*': tokens.push_back({TokenKind::Star, "*"}); ++i; continue;
            case '/': tokens.push_back({TokenKind::Slash, "/"}); ++i; continue;
            case '>': tokens.push_back({TokenKind::Greater, ">"}); ++i; continue;
            case '<': tokens.push_back({TokenKind::Less, "<"}); ++i; continue;
            case ',': tokens.push_back({TokenKind::Comma, ","}); ++i; continue;
            default: break;
        }

        if (is_identifier_start(c)) {
            std::size_t start = i;
            ++i;
            while (i < n && is_identifier_body(input[i])) { ++i; }
            tokens.push_back({TokenKind::Identifier, std::string(input.substr(start, i - start))});
            continue;
        }

        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            std::size_t start = i;
            bool seen_exponent = false;
            bool seen_dot = (c == '.');
            ++i;
            while (i < n) {
                char ch = input[i];
                if (std::isdigit(static_cast<unsigned char>(ch))) { ++i; continue; }
                if (ch == '.' && !seen_dot) { seen_dot = true; ++i; continue; }
                if ((ch == 'e' || ch == 'E') && !seen_exponent) {
                    seen_exponent = true;
                    ++i;
                    if (i < n && (input[i] == '+' || input[i] == '-')) { ++i; }
                    continue;
                }
                break;
            }
            tokens.push_back({TokenKind::Number, std::string(input.substr(start, i - start))});
            continue;
        }

        throw std::runtime_error("Unexpected character in expression: " + std::string(1, c));
    }

    tokens.push_back({TokenKind::End, ""});
    return tokens;
}

int precedence(TokenKind kind) {
    switch (kind) {
        case TokenKind::Question: return 1; // ternary
        case TokenKind::OrOr: return 2;
        case TokenKind::AndAnd: return 3;
        case TokenKind::EqualEqual: return 4;
        case TokenKind::Greater:
        case TokenKind::Less:
        case TokenKind::GreaterEqual:
        case TokenKind::LessEqual: return 5;
        case TokenKind::Plus:
        case TokenKind::Minus: return 6;
        case TokenKind::ShiftLeft:
        case TokenKind::ShiftRight: return 7;
        case TokenKind::Star:
        case TokenKind::Slash: return 8;
        default: return -1;
    }
}

class Parser {
public:
    Parser(const std::string& input, const std::vector<gamax::data_type>& column_types)
        : _tokens(tokenize(input)), _column_types(column_types) {}

    AstNode parse() {
        auto expr = parse_expression(0);
        expect(TokenKind::End);
        return expr;
    }

private:
    AstNode parse_expression(int min_prec) {
        auto lhs = parse_prefix();

        while (true) {
            Token tok = peek();
            int prec = precedence(tok.kind);
            if (prec < min_prec) {
                break;
            }

            if (tok.kind == TokenKind::Question) {
                consume();
                auto true_expr = parse_expression(0);
                expect(TokenKind::Colon);
                auto false_expr = parse_expression(prec);

                AstNode node;
                node.kind = AstNodeKind::Select;
                node.result_type = true_expr.result_type;
                node.lhs = std::make_unique<AstNode>(std::move(lhs));
                node.rhs = std::make_unique<AstNode>(std::move(true_expr));
                node.third = std::make_unique<AstNode>(std::move(false_expr));
                lhs = std::move(node);
                continue;
            }

            auto op_kind = tok.kind;
            consume();
            auto rhs = parse_expression(prec + 1);

            if (op_kind == TokenKind::ShiftLeft || op_kind == TokenKind::ShiftRight) {
                if (rhs.kind != AstNodeKind::Literal) {
                    throw std::runtime_error("Shift amount must be a literal integer");
                }
                int bits = extract_shift_bits(rhs.literal_value, rhs.result_type);
                AstNode node;
                node.kind = (op_kind == TokenKind::ShiftLeft) ? AstNodeKind::ShiftLeft : AstNodeKind::ShiftRight;
                node.result_type = lhs.result_type;
                node.shift_bits = bits;
                node.lhs = std::make_unique<AstNode>(std::move(lhs));
                lhs = std::move(node);
                continue;
            }

            AstNode node;
            node.kind = AstNodeKind::Binary;
            node.binary_op = to_binary_op(op_kind);
            node.result_type = result_type_for_binary(node.binary_op, lhs.result_type, rhs.result_type);
            node.lhs = std::make_unique<AstNode>(std::move(lhs));
            node.rhs = std::make_unique<AstNode>(std::move(rhs));
            lhs = std::move(node);
        }

        return lhs;
    }

    AstNode parse_prefix() {
        Token tok = peek();
        if (tok.kind == TokenKind::Minus) {
            consume();
            auto child = parse_expression(9); // higher than binary ops
            AstNode node;
            node.kind = AstNodeKind::Unary;
            node.unary_op = AstUnaryOp::Negate;
            node.result_type = child.result_type;
            node.lhs = std::make_unique<AstNode>(std::move(child));
            return node;
        }
        return parse_primary();
    }

    AstNode parse_primary() {
        Token tok = peek();
        if (tok.kind == TokenKind::Number) {
            consume();
            auto literal = make_numeric_literal(tok.text, std::nullopt);
            AstNode node;
            node.kind = AstNodeKind::Literal;
            node.literal_value = literal.value;
            node.result_type = literal.type;
            return node;
        }

        if (tok.kind == TokenKind::Identifier) {
            auto id = to_lower(tok.text);

            // Bool literals
            if (id == "true" || id == "false") {
                consume();
                bool value = (id == "true");
                AstNode node;
                node.kind = AstNodeKind::Literal;
                node.literal_value = gamax::scalar(value);
                node.result_type = kBoolType;
                return node;
            }

            // Type cast of numeric literal: type(value)
            if (is_type_name(id) && peek(1).kind == TokenKind::LParen) {
                consume(); // identifier
                consume(TokenKind::LParen);

                bool negative = false;
                if (peek().kind == TokenKind::Minus) {
                    negative = true;
                    consume();
                }

                Token num_tok = expect(TokenKind::Number);
                std::string literal_text = negative ? std::string("-") + num_tok.text : num_tok.text;
                auto literal = make_numeric_literal(literal_text, type_from_name(id));
                consume(TokenKind::RParen);

                AstNode node;
                node.kind = AstNodeKind::Literal;
                node.literal_value = literal.value;
                node.result_type = literal.type;
                return node;
            }

            // Function calls with a single argument
            if (is_function_name(id)) {
                consume();
                consume(TokenKind::LParen);
                auto arg = parse_expression(0);
                consume(TokenKind::RParen);

                AstNode node;
                node.kind = AstNodeKind::Unary;
                node.unary_op = to_unary_op(id);
                node.result_type = arg.result_type;
                node.lhs = std::make_unique<AstNode>(std::move(arg));
                return node;
            }

            if (id.rfind("col", 0) == 0) {
                consume();
                auto index_text = id.substr(3);
                if (index_text.empty()) {
                    throw std::runtime_error("Column reference must include an index, e.g., col0");
                }
                std::size_t idx = static_cast<std::size_t>(std::stoul(index_text));
                if (idx >= _column_types.size()) {
                    throw std::runtime_error("Column index out of range: " + std::to_string(idx));
                }
                AstNode node;
                node.kind = AstNodeKind::ColumnRef;
                node.column_index = static_cast<gamax::size_type>(idx);
                node.result_type = _column_types[idx];
                return node;
            }

            throw std::runtime_error("Unknown identifier: " + tok.text);
        }

        if (tok.kind == TokenKind::LParen) {
            consume();
            auto node = parse_expression(0);
            consume(TokenKind::RParen);
            return node;
        }

        throw std::runtime_error("Unexpected token while parsing expression");
    }

    AstBinaryOp to_binary_op(TokenKind kind) {
        switch (kind) {
            case TokenKind::Plus: return AstBinaryOp::Add;
            case TokenKind::Minus: return AstBinaryOp::Subtract;
            case TokenKind::Star: return AstBinaryOp::Multiply;
            case TokenKind::Slash: return AstBinaryOp::Divide;
            case TokenKind::Greater: return AstBinaryOp::GreaterThan;
            case TokenKind::Less: return AstBinaryOp::LessThan;
            case TokenKind::GreaterEqual: return AstBinaryOp::GreaterEqual;
            case TokenKind::LessEqual: return AstBinaryOp::LessEqual;
            case TokenKind::EqualEqual: return AstBinaryOp::Equal;
            case TokenKind::AndAnd: return AstBinaryOp::And;
            case TokenKind::OrOr: return AstBinaryOp::Or;
            default:
                throw std::runtime_error("Unsupported binary operator");
        }
    }

    AstUnaryOp to_unary_op(std::string_view name) {
        if (name == "sqrt") return AstUnaryOp::Sqrt;
        if (name == "log") return AstUnaryOp::Log;
        if (name == "exp") return AstUnaryOp::Exp;
        throw std::runtime_error("Unknown unary function: " + std::string(name));
    }

    gamax::data_type result_type_for_binary(AstBinaryOp op, gamax::data_type lhs_type, gamax::data_type rhs_type) {
        switch (op) {
            case AstBinaryOp::Add:
            case AstBinaryOp::Subtract:
            case AstBinaryOp::Multiply:
            case AstBinaryOp::Divide:
                return lhs_type;
            case AstBinaryOp::GreaterThan:
            case AstBinaryOp::LessThan:
            case AstBinaryOp::GreaterEqual:
            case AstBinaryOp::LessEqual:
            case AstBinaryOp::Equal:
            case AstBinaryOp::And:
            case AstBinaryOp::Or:
                return kBoolType;
            default:
                return lhs_type;
        }
    }

    Token peek(std::size_t offset = 0) const {
        return _tokens[_pos + offset];
    }

    Token consume() {
        return _tokens[_pos++];
    }

    Token consume(TokenKind expected) {
        Token tok = peek();
        if (tok.kind != expected) {
            throw std::runtime_error("Unexpected token");
        }
        return consume();
    }

    Token expect(TokenKind expected) {
        Token tok = peek();
        if (tok.kind != expected) {
            throw std::runtime_error("Unexpected token");
        }
        return consume();
    }

    static int extract_shift_bits(const gamax::scalar& value, gamax::data_type type) {
        switch (type.id()) {
            case gamax::type_id::INT8: return static_cast<int>(value.get<int8_t>());
            case gamax::type_id::INT16: return static_cast<int>(value.get<int16_t>());
            case gamax::type_id::INT32: return static_cast<int>(value.get<int32_t>());
            case gamax::type_id::INT64: return static_cast<int>(value.get<int64_t>());
            default:
                throw std::runtime_error("Shift amount must be an integer literal");
        }
    }

    std::vector<Token> _tokens;
    std::size_t _pos {0};
    const std::vector<gamax::data_type>& _column_types;
};

} // namespace

AstNode parse_expression_to_ast(const std::string& expr, const std::vector<gamax::data_type>& column_types) {
    Parser parser(expr, column_types);
    return parser.parse();
}

ExprTree ast_to_expr_tree(const AstNode& node) {
    switch (node.kind) {
        case AstNodeKind::Literal:
            return ExprTree::constant(node.literal_value);
        case AstNodeKind::ColumnRef:
            return ExprTree::column(node.column_index, node.result_type);
        case AstNodeKind::Unary: {
            auto child = ast_to_expr_tree(*node.lhs);
            switch (node.unary_op) {
                case AstUnaryOp::Negate: return ExprTree::negate(std::move(child));
                case AstUnaryOp::Sqrt: return ExprTree::sqrt(std::move(child));
                case AstUnaryOp::Log: return ExprTree::log(std::move(child));
                case AstUnaryOp::Exp: return ExprTree::exp(std::move(child));
            }
        }
        case AstNodeKind::Binary: {
            auto left = ast_to_expr_tree(*node.lhs);
            auto right = ast_to_expr_tree(*node.rhs);
            switch (node.binary_op) {
                case AstBinaryOp::Add: return ExprTree::add(std::move(left), std::move(right));
                case AstBinaryOp::Subtract: return ExprTree::subtract(std::move(left), std::move(right));
                case AstBinaryOp::Multiply: return ExprTree::multiply(std::move(left), std::move(right));
                case AstBinaryOp::Divide: return ExprTree::divide(std::move(left), std::move(right));
                case AstBinaryOp::GreaterThan: return ExprTree::greater_than(std::move(left), std::move(right));
                case AstBinaryOp::LessThan: return ExprTree::less_than(std::move(left), std::move(right));
                case AstBinaryOp::GreaterEqual: return ExprTree::greater_equal(std::move(left), std::move(right));
                case AstBinaryOp::LessEqual: return ExprTree::less_equal(std::move(left), std::move(right));
                case AstBinaryOp::Equal: return ExprTree::equal(std::move(left), std::move(right));
                case AstBinaryOp::And: return ExprTree::logical_and(std::move(left), std::move(right));
                case AstBinaryOp::Or: return ExprTree::logical_or(std::move(left), std::move(right));
            }
        }
        case AstNodeKind::Select: {
            auto cond = ast_to_expr_tree(*node.lhs);
            auto true_expr = ast_to_expr_tree(*node.rhs);
            auto false_expr = ast_to_expr_tree(*node.third);
            return ExprTree::select(std::move(cond), std::move(true_expr), std::move(false_expr));
        }
        case AstNodeKind::ShiftLeft: {
            auto child = ast_to_expr_tree(*node.lhs);
            return ExprTree::left_shift(std::move(child), node.shift_bits);
        }
        case AstNodeKind::ShiftRight: {
            auto child = ast_to_expr_tree(*node.lhs);
            return ExprTree::right_shift(std::move(child), node.shift_bits);
        }
    }

    throw std::runtime_error("Unhandled AST node kind");
}

ExprTree parse_expression_to_tree(
    const std::string& expr,
    const std::vector<gamax::data_type>& column_types) {
    auto ast = parse_expression_to_ast(expr, column_types);
    return ast_to_expr_tree(ast);
}

} // namespace gamax::expression
