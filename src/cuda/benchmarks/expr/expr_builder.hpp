#pragma once

#include <string>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <exception>
#include <fstream>
#include <type_traits>

#include "operators/select/select.hpp"
#include "expr/expr_fast.hpp"
#include "data_gen/table_gen.hpp"
#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"

using namespace gamax::expression;

// Base class for expression builders
template<typename KeyT>
struct ExpressionBuilder {
    virtual ~ExpressionBuilder() = default;
    virtual ExprTree operator()(gamax::table_view table) const = 0;
    virtual std::string to_string() const = 0;
    virtual int num_columns() const = 0;
};

// Single column: 3 * col0
template<typename KeyT>
struct SimpleMultiply : public ExpressionBuilder<KeyT> {
    KeyT factor;

    SimpleMultiply(KeyT f) : factor(f) {}

    ExprTree operator()(gamax::table_view table) const override {
        return ExprTree::multiply(
            ExprTree::constant(gamax::scalar(factor)),
            ExprTree::column(0, table[0].type())
        );
    }

    std::string to_string() const override {
        return std::to_string(factor) + " * col0";
    }

    int num_columns() const override { return 1; }
};

// Two columns: col0 * (1 - col1)
template<typename KeyT>
struct TwoColumnMultiply : public ExpressionBuilder<KeyT> {
    TwoColumnMultiply() {}

    ExprTree operator()(gamax::table_view table) const override {
        return ExprTree::multiply(
            ExprTree::column(0, table[0].type()),
            ExprTree::subtract(
                ExprTree::constant(gamax::scalar(static_cast<KeyT>(1))),
                ExprTree::column(1, table[1].type())
            )
        );
    }

    std::string to_string() const override {
        return "col0 * (1 - col1)";
    }

    int num_columns() const override { return 2; }
};

// Three columns: col0 * (1 - col1) * (1 + col2)
template<typename KeyT>
struct ThreeColumnMultiply : public ExpressionBuilder<KeyT> {
    ThreeColumnMultiply() {}

    ExprTree operator()(gamax::table_view table) const override {
        return ExprTree::multiply(
            ExprTree::multiply(
                ExprTree::column(0, table[0].type()),
                ExprTree::subtract(
                    ExprTree::constant(gamax::scalar(static_cast<KeyT>(1))),
                    ExprTree::column(1, table[1].type())
                )
            ),
            ExprTree::add(
                ExprTree::constant(gamax::scalar(static_cast<KeyT>(1))),
                ExprTree::column(2, table[2].type())
            )
        );
    }

    std::string to_string() const override {
        return "col0 * (1 - col1) * (1 + col2)";
    }

    int num_columns() const override { return 3; }
};

// Four columns: col0 * (1 - col1) - col2 * col3
template<typename KeyT>
struct FourColumnArithmetic : public ExpressionBuilder<KeyT> {
    FourColumnArithmetic() {}

    ExprTree operator()(gamax::table_view table) const override {
        return ExprTree::subtract(
            ExprTree::multiply(
                ExprTree::column(0, table[0].type()),
                ExprTree::subtract(
                    ExprTree::constant(gamax::scalar(static_cast<KeyT>(1))),
                    ExprTree::column(1, table[1].type())
                )
            ),
            ExprTree::multiply(
                ExprTree::column(2, table[2].type()),
                ExprTree::column(3, table[3].type())
            )
        );
    }

    std::string to_string() const override {
        return "col0 * (1 - col1) - col2 * col3";
    }

    int num_columns() const override { return 4; }
};

// Two columns And Or expression: (col0 == 0 and col1 == 1) or (col0 == 1 and col1 == 0)
template<typename KeyT>
struct TwoColumnAndOr : public ExpressionBuilder<KeyT> {
    TwoColumnAndOr() {}

    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();

        return ExprTree::logical_or(
            ExprTree::logical_and(
                ExprTree::equal(
                    ExprTree::column(0, col0_type),
                    ExprTree::constant(gamax::scalar(static_cast<KeyT>(0)))
                ),
                ExprTree::equal(
                    ExprTree::column(1, col1_type),
                    ExprTree::constant(gamax::scalar(static_cast<KeyT>(1)))
                )
            ),
            ExprTree::logical_and(
                ExprTree::equal(
                    ExprTree::column(0, col0_type),
                    ExprTree::constant(gamax::scalar(static_cast<KeyT>(1)))
                ),
                ExprTree::equal(
                    ExprTree::column(1, col1_type),
                    ExprTree::constant(gamax::scalar(static_cast<KeyT>(0)))
                )
            )
        );
    }

    std::string to_string() const override {
        return "(col0 == 0 AND col1 == 1) OR (col0 == 1 AND col1 == 0)";
    }

    int num_columns() const override { return 2; }
};

// Single column: col0 > threshold
template<typename KeyT>
struct SimpleGreaterThan : public ExpressionBuilder<KeyT> {
    KeyT threshold;
    
    constexpr SimpleGreaterThan(KeyT thresh) : threshold(thresh) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        return ExprTree::greater_than(
            ExprTree::column(0, table[0].type()),
            ExprTree::constant(gamax::scalar(threshold))
        );
    }
    
    std::string to_string() const override {
        return "col0 > " + std::to_string(threshold);
    }
    
    int num_columns() const override { return 1; }
};

// Single column: col0 >= threshold
template<typename KeyT>
struct SimpleGreaterEqual : public ExpressionBuilder<KeyT> {
    KeyT threshold;
    
    SimpleGreaterEqual(KeyT thresh) : threshold(thresh) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        return ExprTree::greater_equal(
            ExprTree::column(0, table[0].type()),
            ExprTree::constant(gamax::scalar(threshold))
        );
    }
    
    std::string to_string() const override {
        return "col0 >= " + std::to_string(threshold);
    }
    
    int num_columns() const override { return 1; }
};

// Single column: col0 < threshold
template<typename KeyT>
struct SimpleLessThan : public ExpressionBuilder<KeyT> {
    KeyT threshold;
    
    SimpleLessThan(KeyT thresh) : threshold(thresh) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        return ExprTree::less_than(
            ExprTree::column(0, table[0].type()),
            ExprTree::constant(gamax::scalar(threshold))
        );
    }
    
    std::string to_string() const override {
        return "col0 < " + std::to_string(threshold);
    }
    
    int num_columns() const override { return 1; }
};

// Single column range: col0 > low AND col0 < high
template<typename KeyT>
struct RangeAndExpr : public ExpressionBuilder<KeyT> {
    KeyT low, high;
    
    RangeAndExpr(KeyT l, KeyT h) : low(l), high(h) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col_type = table[0].type();
        return ExprTree::logical_and(
            ExprTree::greater_than(
                ExprTree::column(0, col_type),
                ExprTree::constant(gamax::scalar(low))
            ),
            ExprTree::less_than(
                ExprTree::column(0, col_type),
                ExprTree::constant(gamax::scalar(high))
            )
        );
    }
    
    std::string to_string() const override {
        return "col0 > " + std::to_string(low) + " AND col0 < " + std::to_string(high);
    }
    
    int num_columns() const override { return 1; }
};

// Single column: col0 < low OR col0 > high
template<typename KeyT>
struct RangeOrExpr : public ExpressionBuilder<KeyT> {
    KeyT low, high;
    
    RangeOrExpr(KeyT l, KeyT h) : low(l), high(h) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col_type = table[0].type();
        return ExprTree::logical_or(
            ExprTree::less_than(
                ExprTree::column(0, col_type),
                ExprTree::constant(gamax::scalar(low))
            ),
            ExprTree::greater_than(
                ExprTree::column(0, col_type),
                ExprTree::constant(gamax::scalar(high))
            )
        );
    }
    
    std::string to_string() const override {
        return "col0 < " + std::to_string(low) + " OR col0 > " + std::to_string(high);
    }
    
    int num_columns() const override { return 1; }
};

// Two columns: col0 + col1 > threshold
template<typename KeyT>
struct TwoColumnSum : public ExpressionBuilder<KeyT> {
    KeyT threshold;
    
    TwoColumnSum(KeyT thresh) : threshold(thresh) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();
        
        return ExprTree::greater_than(
            ExprTree::add(
                ExprTree::column(0, col0_type),
                ExprTree::column(1, col1_type)
            ),
            ExprTree::constant(gamax::scalar(threshold))
        );
    }
    
    std::string to_string() const override {
        return "col0 + col1 > " + std::to_string(threshold);
    }
    
    int num_columns() const override { return 2; }
};

// Two columns: col0 > threshold1 AND col1 < threshold2
template<typename KeyT>
struct TwoColumnAndExpr : public ExpressionBuilder<KeyT> {
    KeyT threshold1, threshold2;
    
    TwoColumnAndExpr(KeyT t1, KeyT t2) : threshold1(t1), threshold2(t2) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();
        
        return ExprTree::logical_and(
            ExprTree::greater_than(
                ExprTree::column(0, col0_type),
                ExprTree::constant(gamax::scalar(threshold1))
            ),
            ExprTree::less_than(
                ExprTree::column(1, col1_type),
                ExprTree::constant(gamax::scalar(threshold2))
            )
        );
    }
    
    std::string to_string() const override {
        return "col0 > " + std::to_string(threshold1) + " AND col1 < " + std::to_string(threshold2);
    }
    
    int num_columns() const override { return 2; }
};

// Two columns: col0 <= threshold1 OR col1 >= threshold2
template<typename KeyT>
struct TwoColumnOrExpr : public ExpressionBuilder<KeyT> {
    KeyT threshold1, threshold2;
    
    TwoColumnOrExpr(KeyT t1, KeyT t2) : threshold1(t1), threshold2(t2) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();
        
        return ExprTree::logical_or(
            ExprTree::less_equal(
                ExprTree::column(0, col0_type),
                ExprTree::constant(gamax::scalar(threshold1))
            ),
            ExprTree::greater_equal(
                ExprTree::column(1, col1_type),
                ExprTree::constant(gamax::scalar(threshold2))
            )
        );
    }
    
    std::string to_string() const override {
        return "col0 <= " + std::to_string(threshold1) + " OR col1 >= " + std::to_string(threshold2);
    }
    
    int num_columns() const override { return 2; }
};

// Three columns: (col0 + col1) * col2 > threshold
template<typename KeyT>
struct ThreeColumnComplexExpr : public ExpressionBuilder<KeyT> {
    KeyT threshold;
    
    ThreeColumnComplexExpr(KeyT thresh) : threshold(thresh) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();
        auto col2_type = table[2].type();
        
        return ExprTree::greater_than(
            ExprTree::multiply(
                ExprTree::add(
                    ExprTree::column(0, col0_type),
                    ExprTree::column(1, col1_type)
                ),
                ExprTree::column(2, col2_type)
            ),
            ExprTree::constant(gamax::scalar(threshold))
        );
    }
    
    std::string to_string() const override {
        return "(col0 + col1) * col2 > " + std::to_string(threshold);
    }
    
    int num_columns() const override { return 3; }
};

// Three columns: col0 > t1 AND col1 < t2 AND col2 >= t3
template<typename KeyT>
struct ThreeColumnAndExpr : public ExpressionBuilder<KeyT> {
    KeyT t1, t2, t3;
    
    ThreeColumnAndExpr(KeyT threshold1, KeyT threshold2, KeyT threshold3) 
        : t1(threshold1), t2(threshold2), t3(threshold3) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();
        auto col2_type = table[2].type();
        
        return ExprTree::logical_and(
            ExprTree::logical_and(
                ExprTree::greater_than(
                    ExprTree::column(0, col0_type),
                    ExprTree::constant(gamax::scalar(t1))
                ),
                ExprTree::less_than(
                    ExprTree::column(1, col1_type),
                    ExprTree::constant(gamax::scalar(t2))
                )
            ),
            ExprTree::greater_equal(
                ExprTree::column(2, col2_type),
                ExprTree::constant(gamax::scalar(t3))
            )
        );
    }
    
    std::string to_string() const override {
        return "col0 > " + std::to_string(t1) + " AND col1 < " + std::to_string(t2) + 
               " AND col2 >= " + std::to_string(t3);
    }
    
    int num_columns() const override { return 3; }
};

// Four columns: (col0 + col1) == (col2 + col3)
template<typename KeyT>
struct FourColumnEqualExpr : public ExpressionBuilder<KeyT> {
    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();
        auto col2_type = table[2].type();
        auto col3_type = table[3].type();
        
        return ExprTree::equal(
            ExprTree::add(
                ExprTree::column(0, col0_type),
                ExprTree::column(1, col1_type)
            ),
            ExprTree::add(
                ExprTree::column(2, col2_type),
                ExprTree::column(3, col3_type)
            )
        );
    }
    
    std::string to_string() const override {
        return "(col0 + col1) == (col2 + col3)";
    }
    
    int num_columns() const override { return 4; }
};

// Operator overloading example: col0 + col1 > threshold
template<typename KeyT>
struct TwoColumnSumOperator : public ExpressionBuilder<KeyT> {
    KeyT threshold;
    
    TwoColumnSumOperator(KeyT thresh) : threshold(thresh) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();
        
        return (ExprTree::column(0, col0_type) + ExprTree::column(1, col1_type)) 
             > ExprTree::constant(gamax::scalar(threshold));
    }
    
    std::string to_string() const override {
        return "col0 + col1 > " + std::to_string(threshold) + " (using operators)";
    }
    
    int num_columns() const override { return 2; }
};

// Operator overloading: col0 > t1 && col0 < t2
template<typename KeyT>
struct RangeAndOperator : public ExpressionBuilder<KeyT> {
    KeyT low, high;
    
    RangeAndOperator(KeyT l, KeyT h) : low(l), high(h) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col_type = table[0].type();
        auto col = ExprTree::column(0, col_type);
        
        return (col > ExprTree::constant(gamax::scalar(low))) 
            && (col < ExprTree::constant(gamax::scalar(high)));
    }
    
    std::string to_string() const override {
        return "col0 > " + std::to_string(low) + " && col0 < " + std::to_string(high) + " (using operators)";
    }
    
    int num_columns() const override { return 1; }
};

// Operator overloading: (col0 + col1) * col2 > threshold
template<typename KeyT>
struct ThreeColumnComplexOperator : public ExpressionBuilder<KeyT> {
    KeyT threshold;
    
    ThreeColumnComplexOperator(KeyT thresh) : threshold(thresh) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col0_type = table[0].type();
        auto col1_type = table[1].type();
        auto col2_type = table[2].type();
        
        return ((ExprTree::column(0, col0_type) + ExprTree::column(1, col1_type)) 
              * ExprTree::column(2, col2_type)) 
             > ExprTree::constant(gamax::scalar(threshold));
    }
    
    std::string to_string() const override {
        return "(col0 + col1) * col2 > " + std::to_string(threshold) + " (using operators)";
    }
    
    int num_columns() const override { return 3; }
};

// Shift operator example: col0 >> 2 > threshold
template<typename KeyT>
struct ShiftOperator : public ExpressionBuilder<KeyT> {
    KeyT threshold;
    
    ShiftOperator(KeyT thresh) : threshold(thresh) {}
    
    ExprTree operator()(gamax::table_view table) const override {
        auto col_type = table[0].type();
        
        return (ExprTree::column(0, col_type) >> 2) 
             > ExprTree::constant(gamax::scalar(threshold));
    }
    
    std::string to_string() const override {
        return "col0 >> 2 > " + std::to_string(threshold);
    }
    
    int num_columns() const override { return 1; }
};