#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda.h>

#include "operators/select/predicate.hpp"
#include "operators/select/select.hpp"
#include "expr/expr_fast.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"
#include "tests/test_helper.hpp"

namespace {

using namespace gamax::expression;
using namespace gamax::select;

class PredicateTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }

    cudaStream_t stream_{};

    // Helper: create a numeric column from host data
    template<typename T>
    std::unique_ptr<gamax::column> make_numeric_col(const std::vector<T>& data) {
        return gamax::make_column_from_host(data.data(), data.size());
    }

    // Helper: create a string_column from a vector of std::string
    std::unique_ptr<gamax::string_column> make_string_col(const std::vector<std::string>& strings) {
        std::vector<char> flat;
        std::vector<int32_t> lens;
        for (const auto& s : strings) {
            lens.push_back(static_cast<int32_t>(s.size()));
            flat.insert(flat.end(), s.begin(), s.end());
        }
        auto ptr = gamax::make_string_column(
            flat.data(), lens.data(),
            static_cast<gamax::size_type>(strings.size()),
            stream_, gamax::ARROW_STRING);
        return std::unique_ptr<gamax::string_column>(
            static_cast<gamax::string_column*>(ptr.release()));
    }

    // Helper: read a BOOL8 bitmask column back to host
    std::vector<bool> bitmask_to_host(gamax::column_view col) {
        EXPECT_EQ(col.type().id(), gamax::type_id::BOOL8);
        auto host = gamax::copy_column_to_host<bool>(col, stream_);
        cudaStreamSynchronize(stream_);
        std::vector<bool> result(col.size());
        for (gamax::size_type i = 0; i < col.size(); ++i) {
            result[i] = host[i];
        }
        return result;
    }
};

// ────────────────────────────────────────────────────────────────────────────
// 1. Pure numeric ExprTree wrapped in a Predicate
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, NumericExprOnly) {
    // Table: col0 = [1, 5, 10, 15, 20]
    auto col0 = make_numeric_col<int32_t>({1, 5, 10, 15, 20});
    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));

    // Predicate: col0 > 8
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::greater_than(
        ExprTree::column(0, int_type),
        ExprTree::constant(gamax::scalar(8))
    );
    auto pred = Predicate::expr(std::move(expr));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    std::vector<bool> expected = {false, false, true, true, true};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 2. Pure string predicate: string_equal
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, StringEqualOnly) {
    // Table: col0 (int, dummy), col1 (string)
    auto col0 = make_numeric_col<int32_t>({1, 2, 3, 4, 5});
    auto col1 = make_string_col({"apple", "banana", "apple", "cherry", "apple"});

    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));
    table->add_column(std::move(col1));

    auto scalar = gamax::make_string_scalar("apple", stream_);
    auto pred = Predicate::string_equal(
        1, std::move(scalar));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    std::vector<bool> expected = {true, false, true, false, true};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 3. Pure string predicate: string_startswith
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, StringStartswithOnly) {
    auto col0 = make_string_col({"hello", "world", "help", "hero", "heap"});
    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));

    auto scalar = gamax::make_string_scalar("he", stream_);
    auto pred = Predicate::string_startswith(0, std::move(scalar));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    std::vector<bool> expected = {true, false, true, true, true};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 4. Pure string predicate: string_endswith
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, StringEndswithOnly) {
    auto col0 = make_string_col({"running", "jumping", "sitting", "read", "sleeping"});
    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));

    auto scalar = gamax::make_string_scalar("ing", stream_);
    auto pred = Predicate::string_endswith(0, std::move(scalar));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    std::vector<bool> expected = {true, true, true, false, true};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 5. Pure string predicate: string_contains
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, StringContainsOnly) {
    auto col0 = make_string_col({"abcdef", "ghijkl", "abxyz", "xyzabc", "nope"});
    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));

    auto scalar = gamax::make_string_scalar("abc", stream_);
    auto pred = Predicate::string_contains(0, std::move(scalar));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    std::vector<bool> expected = {true, false, false, true, false};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 6. Mixed: numeric AND string_equal
//    col0 > 2 AND col1 == "apple"
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, NumericAndStringEqual) {
    // col0: [1, 2, 3, 4, 5]
    // col1: ["apple", "banana", "apple", "cherry", "apple"]
    auto col0 = make_numeric_col<int32_t>({1, 2, 3, 4, 5});
    auto col1 = make_string_col({"apple", "banana", "apple", "cherry", "apple"});

    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));
    table->add_column(std::move(col1));

    // Numeric: col0 > 2
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto numeric_pred = Predicate::expr(
        ExprTree::greater_than(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(2))
        )
    );

    // String: col1 == "apple"
    auto scalar = gamax::make_string_scalar("apple", stream_);
    auto string_pred = Predicate::string_equal(1, std::move(scalar));

    // Combined: numeric AND string
    auto pred = Predicate::logical_and(std::move(numeric_pred), std::move(string_pred));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    // col0>2:    [F, F, T, T, T]
    // col1=apple:[T, F, T, F, T]
    // AND:       [F, F, T, F, T]
    std::vector<bool> expected = {false, false, true, false, true};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 7. Mixed: numeric OR string_startswith
//    col0 < 3 OR startswith(col1, "ch")
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, NumericOrStringStartswith) {
    auto col0 = make_numeric_col<int32_t>({1, 2, 3, 4, 5});
    auto col1 = make_string_col({"apple", "cherry", "banana", "chocolate", "date"});

    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));
    table->add_column(std::move(col1));

    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto numeric_pred = Predicate::expr(
        ExprTree::less_than(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(3))
        )
    );

    auto scalar = gamax::make_string_scalar("ch", stream_);
    auto string_pred = Predicate::string_startswith(1, std::move(scalar));

    auto pred = Predicate::logical_or(std::move(numeric_pred), std::move(string_pred));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    // col0<3:         [T, T, F, F, F]
    // startswith "ch": [F, T, F, T, F]
    // OR:             [T, T, F, T, F]
    std::vector<bool> expected = {true, true, false, true, false};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 8. NOT predicate
//    NOT(col0 == "apple")
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, NotStringEqual) {
    auto col0 = make_string_col({"apple", "banana", "apple", "cherry", "date"});
    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));

    auto scalar = gamax::make_string_scalar("apple", stream_);
    auto string_pred = Predicate::string_equal(0, std::move(scalar));
    auto pred = Predicate::logical_not(std::move(string_pred));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    std::vector<bool> expected = {false, true, false, true, true};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 9. Operator overloads: pred1 && pred2 || !pred3
//    (col0 > 2) && string_equal(col1, "apple") || NOT(string_contains(col1, "ch"))
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, OperatorOverloads) {
    auto col0 = make_numeric_col<int32_t>({1, 2, 3, 4, 5});
    auto col1 = make_string_col({"apple", "cherry", "apple", "chocolate", "date"});

    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));
    table->add_column(std::move(col1));

    auto int_type = gamax::data_type(gamax::type_id::INT32);

    auto p1 = Predicate::expr(
        ExprTree::greater_than(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(2))
        )
    );

    auto s_eq = gamax::make_string_scalar("apple", stream_);
    auto p2 = Predicate::string_equal(1, std::move(s_eq));

    auto s_ch = gamax::make_string_scalar("ch", stream_);
    auto p3 = Predicate::string_contains(1, std::move(s_ch));

    // (col0 > 2 && col1 == "apple") || NOT(contains(col1, "ch"))
    auto pred = (p1 && p2) || !p3;

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    // p1 (col0>2):        [F, F, T, T, T]
    // p2 (col1="apple"):  [T, F, T, F, F]
    // p1 && p2:           [F, F, T, F, F]
    // p3 (contains "ch"): [F, T, F, T, F]
    // !p3:                [T, F, T, F, T]
    // final OR:           [T, F, T, F, T]
    std::vector<bool> expected = {true, false, true, false, true};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 10. Complex: (col0 >= 3 AND col0 <= 8) AND (startswith(col1, "a") OR endswith(col1, "e"))
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, ComplexMixedPredicate) {
    auto col0 = make_numeric_col<int32_t>({1, 3, 5, 7, 9});
    auto col1 = make_string_col({"apple", "avocado", "grape", "active", "banana"});

    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));
    table->add_column(std::move(col1));

    auto int_type = gamax::data_type(gamax::type_id::INT32);

    // col0 >= 3 AND col0 <= 8
    auto numeric_pred = Predicate::expr(
        ExprTree::logical_and(
            ExprTree::greater_equal(
                ExprTree::column(0, int_type),
                ExprTree::constant(gamax::scalar(3))
            ),
            ExprTree::less_equal(
                ExprTree::column(0, int_type),
                ExprTree::constant(gamax::scalar(8))
            )
        )
    );

    // startswith(col1, "a")
    auto s_a = gamax::make_string_scalar("a", stream_);
    auto sw_pred = Predicate::string_startswith(1, std::move(s_a));

    // endswith(col1, "e")
    auto s_e = gamax::make_string_scalar("e", stream_);
    auto ew_pred = Predicate::string_endswith(1, std::move(s_e));

    // (numeric) AND (startswith OR endswith)
    auto pred = numeric_pred && (sw_pred || ew_pred);

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    // col0 in [3,8]:           [F, T, T, T, F]
    // startswith(col1, "a"):   [T, T, F, T, F]
    // endswith(col1, "e"):     [T, F, T, T, F]
    // sw OR ew:                [T, T, T, T, F]
    // final AND:               [F, T, T, T, F]
    std::vector<bool> expected = {false, true, true, true, false};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 11. string_in — exact set-membership against a small value list
// ────────────────────────────────────────────────────────────────────────────
//
// The previous revision of this test was written when STRING_IN was a
// `throw "not yet implemented"` stub in predicate.cu; it asserted the
// throw, and grew stale once `gamax::in_multiple` landed. Positive
// coverage here: distinguish *exact match* (x IN (...)) from prefix-
// match and substring-match, and pin down the empty-list edge case.

TEST_F(PredicateTest, StringInMatchesSetMembership) {
    // Inputs chosen so the bitmask is distinguishable from what you'd
    // get with alternative semantics:
    //   - "ab" rules out "prefix match" (would match "a").
    //   - "ab" also rules out "contains"   (would match both "a" and "b").
    //   - "aa" rules out "same length wins" (would match "a" or "b").
    auto col0 = make_string_col({"a", "b", "c", "ab", "aa"});
    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));

    auto s1 = gamax::make_string_scalar("a", stream_);
    auto s2 = gamax::make_string_scalar("b", stream_);
    std::vector<std::shared_ptr<gamax::string_scalar>> values;
    values.push_back(std::move(s1));
    values.push_back(std::move(s2));

    auto pred = Predicate::string_in(0, std::move(values));
    auto mask = gamax::select::select_to_bitmask(
        table->view(), pred, stream_);
    auto result = bitmask_to_host(mask->view());
    // IN ('a','b'):  "a"→true, "b"→true, "c"/"ab"/"aa"→false.
    std::vector<bool> expected = {true, true, false, false, false};
    EXPECT_EQ(result, expected);
}

TEST_F(PredicateTest, StringInEmptyValueListIsUniformlyFalse) {
    // SQL semantics: `x IN ()` is always false. Regression pin for the
    // explicit memset-to-zero path documented in the in_multiple
    // landing commit (f18e3fc).
    auto col0 = make_string_col({"a", "b", "c"});
    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));

    std::vector<std::shared_ptr<gamax::string_scalar>> values;  // empty
    auto pred = Predicate::string_in(0, std::move(values));
    auto mask = gamax::select::select_to_bitmask(
        table->view(), pred, stream_);
    auto result = bitmask_to_host(mask->view());
    std::vector<bool> expected = {false, false, false};
    EXPECT_EQ(result, expected);
}

// ────────────────────────────────────────────────────────────────────────────
// 12. string_contains_multiple
// ────────────────────────────────────────────────────────────────────────────

TEST_F(PredicateTest, StringContainsMultiple) {
    auto col0 = make_string_col({"abcdef", "ghijkl", "xyzabc", "mnopqr", "abcxyz"});
    auto table = std::make_unique<gamax::table>();
    table->add_column(std::move(col0));

    auto s1 = gamax::make_string_scalar("abc", stream_);
    auto s2 = gamax::make_string_scalar("xyz", stream_);
    std::vector<std::shared_ptr<gamax::string_scalar>> patterns;
    patterns.push_back(std::move(s1));
    patterns.push_back(std::move(s2));

    auto pred = Predicate::string_contains_multiple(0, std::move(patterns));

    auto mask = gamax::select::select_to_bitmask(table->view(), pred, stream_);

    auto result = bitmask_to_host(mask->view());
    // contains_multiple checks if string contains ALL patterns in order (non-overlapping)
    // patterns: ["abc", "xyz"]
    // "abcdef" has "abc" at 0, then search for "xyz" from 3 -> not found -> false
    // "ghijkl" has neither                                               -> false
    // "xyzabc" search "abc" from 0 -> found at 3, then "xyz" from 6 -> not found -> false
    // "mnopqr" has neither                                               -> false
    // "abcxyz" search "abc" from 0 -> found at 0, then "xyz" from 3 -> found at 3 -> true
    std::vector<bool> expected = {false, false, false, false, true};
    EXPECT_EQ(result, expected);
}

} // namespace
