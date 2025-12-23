#pragma once

#include <memory>
#include <cuda_runtime_api.h>
#include <vector>
#include <functional>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/scalar.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax::expression {
constexpr int MAX_EXPR_REGISTERS = 3;
constexpr int MAX_OPERANDS = 3;
enum ExprId {
    Constant,
    Column,
    Negate,
    LeftShift,
    RightShift,
    Add,
    Substract,
    Multiply,
    Divide,
    Sqrt,
    Log,
    Exp,
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
    Equal,
    And,
    Or,
    Select // given a boolean expression, choose between two expressions
    // e.g.,  Select(cond, expr_true, expr_false)
};

struct Expr {
    ExprId id;
    gamax::data_type type; // result data type

    // The following two fields are populated when generating the expression 
    // program from the expression tree
    int operand_regs[MAX_OPERANDS]; // registers of operands
    int return_reg; // which register to store the result

    struct constant {
        gamax::scalar value;
    };

    struct column {
        gamax::size_type cid; // column id
    };

    struct left_shift {
        int bits;
    };

    struct right_shift {
        int bits;
    };

    struct log {
        int base;
    };

    union {
        constant c;
        column col;
        left_shift lshift;
        right_shift rshift;
        log lg;
    } data;
};

struct ExprProgram {
    Expr* exprs {nullptr};
    gamax::size_type n_exprs {0};

    ExprProgram(Expr* exprs, gamax::size_type n_exprs) : exprs(exprs), n_exprs(n_exprs) {}
};

class ExprTree {
public:
    ExprTree() = default;
    ExprTree(const ExprTree&) = default;
    ExprTree(ExprTree&&) = default;
    ExprTree& operator=(const ExprTree&) = default;
    ExprTree& operator=(ExprTree&&) = default;

    // Factory methods for leaf nodes
    static ExprTree constant(gamax::scalar value) {
        Expr e{};
        e.id = ExprId::Constant;
        e.type = value.type();
        e.data.c.value = value;
        return ExprTree(e, {});
    }
    
    static ExprTree column(gamax::size_type cid, gamax::data_type type) {
        Expr e{};
        e.id = ExprId::Column;
        e.type = type;
        e.data.col.cid = cid;
        return ExprTree(e, {});
    }
    
    // Unary operators
    static ExprTree negate(ExprTree child) {
        auto type = child.type();
        Expr e{};
        e.id = ExprId::Negate;
        e.type = type;
        return ExprTree(e, {std::make_shared<ExprTree>(std::move(child))});
    }
    
    static ExprTree left_shift(ExprTree child, int bits) {
        auto type = child.type();
        Expr e{};
        e.id = ExprId::LeftShift;
        e.type = type;
        e.data.lshift.bits = bits;
        return ExprTree(e, {std::make_shared<ExprTree>(std::move(child))});
    }
    
    static ExprTree right_shift(ExprTree child, int bits) {
        auto type = child.type();
        Expr e{};
        e.id = ExprId::RightShift;
        e.type = type;
        e.data.rshift.bits = bits;
        return ExprTree(e, {std::make_shared<ExprTree>(std::move(child))});
    }
    
    static ExprTree sqrt(ExprTree child) {
        auto type = child.type();
        Expr e{};
        e.id = ExprId::Sqrt;
        e.type = type;
        return ExprTree(e, {std::make_shared<ExprTree>(std::move(child))});
    }
    
    static ExprTree log(ExprTree child) {
        auto type = child.type();
        Expr e{};
        e.id = ExprId::Log;
        e.type = type;
        return ExprTree(e, {std::make_shared<ExprTree>(std::move(child))});
    }
    
    static ExprTree exp(ExprTree child) {
        auto type = child.type();
        Expr e{};
        e.id = ExprId::Exp;
        e.type = type;
        return ExprTree(e, {std::make_shared<ExprTree>(std::move(child))});
    }
    
    // Binary arithmetic operators
    static ExprTree add(ExprTree left, ExprTree right) {
        auto type = left.type();
        Expr e{};
        e.id = ExprId::Add;
        e.type = type;
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    static ExprTree subtract(ExprTree left, ExprTree right) {
        auto type = left.type();
        Expr e{};
        e.id = ExprId::Substract;
        e.type = type;
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    static ExprTree multiply(ExprTree left, ExprTree right) {
        auto type = left.type();
        Expr e{};
        e.id = ExprId::Multiply;
        e.type = type;
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    static ExprTree divide(ExprTree left, ExprTree right) {
        auto type = left.type();
        Expr e{};
        e.id = ExprId::Divide;
        e.type = type;
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    // Comparison operators
    static ExprTree equal(ExprTree left, ExprTree right) {
        Expr e{};
        e.id = ExprId::Equal;
        e.type = gamax::data_type(gamax::type_id::BOOL8);
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    static ExprTree greater_than(ExprTree left, ExprTree right) {
        Expr e{};
        e.id = ExprId::GreaterThan;
        e.type = gamax::data_type(gamax::type_id::BOOL8);
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    static ExprTree less_than(ExprTree left, ExprTree right) {
        Expr e{};
        e.id = ExprId::LessThan;
        e.type = gamax::data_type(gamax::type_id::BOOL8);
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    static ExprTree greater_equal(ExprTree left, ExprTree right) {
        Expr e{};
        e.id = ExprId::GreaterEqual;
        e.type = gamax::data_type(gamax::type_id::BOOL8);
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    static ExprTree less_equal(ExprTree left, ExprTree right) {
        Expr e{};
        e.id = ExprId::LessEqual;
        e.type = gamax::data_type(gamax::type_id::BOOL8);
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    // Logical operators
    static ExprTree logical_and(ExprTree left, ExprTree right) {
        Expr e{};
        e.id = ExprId::And;
        e.type = gamax::data_type(gamax::type_id::BOOL8);
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    static ExprTree logical_or(ExprTree left, ExprTree right) {
        Expr e{};
        e.id = ExprId::Or;
        e.type = gamax::data_type(gamax::type_id::BOOL8);
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(left)),
            std::make_shared<ExprTree>(std::move(right))
        });
    }
    
    // Ternary operator
    static ExprTree select(ExprTree cond, ExprTree true_expr, ExprTree false_expr) {
        auto type = true_expr.type();
        Expr e{};
        e.id = ExprId::Select;
        e.type = type;
        return ExprTree(e, {
            std::make_shared<ExprTree>(std::move(cond)),
            std::make_shared<ExprTree>(std::move(true_expr)),
            std::make_shared<ExprTree>(std::move(false_expr))
        });
    }
    
    // Operator overloading for natural syntax
    ExprTree operator+(const ExprTree& other) const {
        return add(*this, other);
    }
    
    ExprTree operator-(const ExprTree& other) const {
        return subtract(*this, other);
    }
    
    ExprTree operator*(const ExprTree& other) const {
        return multiply(*this, other);
    }
    
    ExprTree operator/(const ExprTree& other) const {
        return divide(*this, other);
    }
    
    ExprTree operator>(const ExprTree& other) const {
        return greater_than(*this, other);
    }
    
    ExprTree operator<(const ExprTree& other) const {
        return less_than(*this, other);
    }
    
    ExprTree operator>=(const ExprTree& other) const {
        return greater_equal(*this, other);
    }
    
    ExprTree operator<=(const ExprTree& other) const {
        return less_equal(*this, other);
    }
    
    ExprTree operator==(const ExprTree& other) const {
        return equal(*this, other);
    }
    
    ExprTree operator&&(const ExprTree& other) const {
        return logical_and(*this, other);
    }
    
    ExprTree operator||(const ExprTree& other) const {
        return logical_or(*this, other);
    }
    
    ExprTree operator-() const {
        return negate(*this);
    }
    
    ExprTree operator<<(int bits) const {
        return left_shift(*this, bits);
    }
    
    ExprTree operator>>(int bits) const {
        return right_shift(*this, bits);
    }

    // Accessors
    Expr& root() {
        return _root;
    }

    const Expr& root() const {
        return _root;
    }

    gamax::data_type type() const {
        return _root.type;
    }

    auto make_program_host() {
        std::vector<Expr> program_exprs;
        
        generate_code_sethi_ullman(*this, 0, program_exprs);

        return program_exprs;
    }

    auto make_program(cudaStream_t stream = cudaStreamDefault) {
        std::vector<Expr> program_exprs = make_program_host();

        Expr* d_exprs;
        gamax::size_type n_exprs = program_exprs.size();
        allocate_mem(&d_exprs, false, n_exprs * sizeof(Expr), stream);
        cudaMemcpyAsync(d_exprs, program_exprs.data(), n_exprs * sizeof(Expr), cudaMemcpyHostToDevice, stream);

        auto deleter = [](ExprProgram* t) { release_mem(t->exprs); };
        return std::unique_ptr<ExprProgram, decltype(deleter)>{
            new ExprProgram(d_exprs, n_exprs), deleter};
    }

    int max_regs() const {
        return _max_regs;
    }

    gamax::type_id get_common_type() const {
        gamax::type_id common_type = gamax::type_id::EMPTY;
        bool first = true;
        
        std::function<void(const ExprTree&)> traverse = [&](const ExprTree& tree) {
            if (tree._root.id == ExprId::Constant || tree._root.id == ExprId::Column) {
                gamax::type_id current_type = tree._root.type.id();
                if (first) {
                    common_type = current_type;
                    first = false;
                } else if (common_type != current_type) {
                    common_type = gamax::type_id::EMPTY;
                }
            }
            
            for (const auto& child : tree._children) {
                if (child) {
                    traverse(*child);
                }
            }
        };
        
        traverse(*this);
        return common_type;
    }

private:
    // Private constructor
    ExprTree(Expr e, std::vector<std::shared_ptr<ExprTree>> children) 
        : _root(e), _children(std::move(children)) {}

    // Helper to calculate Sethi-Ullman numbers (min registers needed)
    int calc_min_regs(const ExprTree& tree) const {
        if (tree._children.empty()) {
            return 1; // Leaf needs 1 register
        }
        
        int left_regs = 0;
        int right_regs = 0;
        
        if (tree._children.size() > 0 && tree._children[0]) {
            left_regs = calc_min_regs(*tree._children[0]);
        }
        if (tree._children.size() > 1 && tree._children[1]) {
            right_regs = calc_min_regs(*tree._children[1]);
        }

        if (left_regs == right_regs) {
            return left_regs + 1;
        } else {
            return std::max(left_regs, right_regs);
        }
    }

    // Helper to generate code based on Sethi-Ullman labeling
    // stack_idx represents the current "base" register available to use
    void generate_code_sethi_ullman(const ExprTree& tree, int stack_idx, std::vector<Expr>& program_exprs) {
        _max_regs = std::max(_max_regs, stack_idx + 1);

        // Leaf node
        if (tree._children.empty()) {
            Expr e = tree._root;
            e.return_reg = stack_idx;
            program_exprs.push_back(e);
            return;
        }

        // Unary operator
        if (tree._children.size() == 1) {
            generate_code_sethi_ullman(*tree._children[0], stack_idx, program_exprs);
            Expr e = tree._root;
            e.operand_regs[0] = stack_idx;
            e.return_reg = stack_idx;
            program_exprs.push_back(e);
            return;
        }

        // Binary operator (Sethi-Ullman logic)
        if (tree._children.size() == 2) {
            int left_regs = calc_min_regs(*tree._children[0]);
            int right_regs = calc_min_regs(*tree._children[1]);

            if (left_regs >= right_regs) {
                // Evaluate left first
                generate_code_sethi_ullman(*tree._children[0], stack_idx, program_exprs);
                // Evaluate right, result goes to stack_idx + 1 (keeping stack_idx for left result)
                generate_code_sethi_ullman(*tree._children[1], stack_idx + 1, program_exprs);
                
                Expr e = tree._root;
                e.operand_regs[0] = stack_idx;     // Left result
                e.operand_regs[1] = stack_idx + 1; // Right result
                e.return_reg = stack_idx;          // Store result in base
                program_exprs.push_back(e);
            } else {
                // Evaluate right first
                generate_code_sethi_ullman(*tree._children[1], stack_idx, program_exprs);
                // Evaluate left, result goes to stack_idx + 1
                generate_code_sethi_ullman(*tree._children[0], stack_idx + 1, program_exprs);

                Expr e = tree._root;
                e.operand_regs[0] = stack_idx + 1; // Left result (evaluated second)
                e.operand_regs[1] = stack_idx;     // Right result (evaluated first)
                e.return_reg = stack_idx;          // Store result in base
                program_exprs.push_back(e);
            }
            return;
        }
        
        // Fallback for ternary (Select) or others: simple left-to-right
        for (size_t i = 0; i < tree._children.size(); ++i) {
            generate_code_sethi_ullman(*tree._children[i], stack_idx + i, program_exprs);
        }
        Expr e = tree._root;
        for (size_t i = 0; i < tree._children.size(); ++i) {
            e.operand_regs[i] = stack_idx + i;
        }
        e.return_reg = stack_idx;
        program_exprs.push_back(e);
    }

    Expr _root;
    int _max_regs {-1};
    std::vector<std::shared_ptr<ExprTree>> _children;
};

std::unique_ptr<gamax::column> evaluate_expr(
    gamax::table_view table,
    ExprTree& et, 
    cudaStream_t stream = cudaStreamDefault);

// one operator per kernel
std::unique_ptr<gamax::column> evaluate_expr_nofuse(
    gamax::table_view table,
    ExprTree& et, 
    cudaStream_t stream = cudaStreamDefault);
} // namespace gamax::expression