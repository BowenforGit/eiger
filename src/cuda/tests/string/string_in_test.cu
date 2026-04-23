// Unit tests for `gamax::in_multiple` — the STRING `IN (p0, p1, ...)`
// predicate. Mirrors the multi-pattern staging used by
// `contains_multiple` (see string_fast.cu), but with equality
// semantics rather than substring containment.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "memory/column.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"

namespace {

// Build an arrow string_column from host strings.
gamax::string_column_ptr make_string_column_from_host(
    const std::vector<std::string>& strs,
    cudaStream_t stream = cudaStreamDefault)
{
    std::string flat;
    std::vector<int32_t> lens;
    lens.reserve(strs.size());
    for (const auto& s : strs) {
        flat.append(s);
        lens.push_back(static_cast<int32_t>(s.size()));
    }
    return gamax::make_string_column(
        flat.data(),
        lens.data(),
        static_cast<gamax::size_type>(strs.size()),
        stream);
}

// Helper that constructs string_scalars for each pattern, runs
// `in_multiple`, and returns the bool result as a host vector.
std::vector<bool> run_in_multiple(
    const std::vector<std::string>& rows,
    const std::vector<std::string>& patterns,
    cudaStream_t stream = cudaStreamDefault)
{
    auto col = make_string_column_from_host(rows, stream);

    std::vector<std::unique_ptr<gamax::string_scalar>> scalars;
    std::vector<gamax::string_scalar_view> views;
    scalars.reserve(patterns.size());
    views.reserve(patterns.size());
    for (const auto& p : patterns) {
        scalars.push_back(gamax::make_string_scalar(p, stream));
        views.push_back(scalars.back()->view());
    }
    auto out = gamax::in_multiple(col->view_s(), views, stream);

    auto host = gamax::copy_column_to_host<bool>(out->view(), stream);
    cudaStreamSynchronize(stream);
    std::vector<bool> result(out->size());
    for (gamax::size_type i = 0; i < out->size(); ++i) result[i] = host[i];
    return result;
}

} // namespace

// ── Basic shape ──

TEST(StringInMultiple, SinglePatternExactMatch) {
    // With one pattern, `in_multiple` should behave exactly like
    // `equal` on a scalar: only rows that *exactly* equal the pattern
    // match, prefixes / suffixes / supersets do not.
    auto got = run_in_multiple(
        {"APPLE", "APPLES", "apple", "APPL", "APPLE"},
        {"APPLE"});
    ASSERT_EQ(got.size(), 5u);
    EXPECT_TRUE(got[0]);
    EXPECT_FALSE(got[1]);
    EXPECT_FALSE(got[2]);
    EXPECT_FALSE(got[3]);
    EXPECT_TRUE(got[4]);
}

TEST(StringInMultiple, MultiplePatterns) {
    auto got = run_in_multiple(
        {"APPLE", "BANANA", "CHERRY", "DATE", "ORANGE", "apple", ""},
        {"APPLE", "BANANA", "ORANGE"});
    ASSERT_EQ(got.size(), 7u);
    EXPECT_TRUE(got[0]);   // APPLE
    EXPECT_TRUE(got[1]);   // BANANA
    EXPECT_FALSE(got[2]);  // CHERRY
    EXPECT_FALSE(got[3]);  // DATE
    EXPECT_TRUE(got[4]);   // ORANGE
    EXPECT_FALSE(got[5]);  // case-sensitive: "apple" != "APPLE"
    EXPECT_FALSE(got[6]);  // empty row is not "APPLE" etc.
}

// ── Length + prefix edge cases ──

TEST(StringInMultiple, DistinguishesByLength) {
    // Rows that are a prefix of a pattern must not match — proves the
    // kernel's length-pre-check isn't skipped.
    auto got = run_in_multiple(
        {"AB", "ABC", "ABCD", "ABCDE"},
        {"ABC", "ABCDE"});
    ASSERT_EQ(got.size(), 4u);
    EXPECT_FALSE(got[0]);
    EXPECT_TRUE(got[1]);
    EXPECT_FALSE(got[2]);
    EXPECT_TRUE(got[3]);
}

TEST(StringInMultiple, EmptyStringRow) {
    // Empty row should only match when "" is one of the patterns.
    {
        auto got = run_in_multiple({"", "x", ""}, {"x"});
        EXPECT_EQ(got, (std::vector<bool>{false, true, false}));
    }
    {
        auto got = run_in_multiple({"", "x", ""}, {"", "x"});
        EXPECT_EQ(got, (std::vector<bool>{true, true, true}));
    }
}

TEST(StringInMultiple, EmptyPatternList) {
    // `x IN ()` is uniformly false (SQL semantics). Every row → false.
    auto got = run_in_multiple({"a", "b", "c", ""}, {});
    ASSERT_EQ(got.size(), 4u);
    for (bool v : got) EXPECT_FALSE(v);
}

// ── Longer strings ──

TEST(StringInMultiple, LongPatternsBeyondInlinedPrefix) {
    // Patterns longer than 32 bytes — exercise the full-string compare
    // path past any inlined prefix that some string representations
    // keep in the header.
    const std::string a(40, 'a');
    const std::string b = std::string(40, 'a').substr(0, 39) + "b";  // differs only at byte 39
    const std::string c(40, 'c');
    auto got = run_in_multiple({a, b, c, std::string(40, 'd')},
                                {a, c});
    EXPECT_EQ(got, (std::vector<bool>{true, false, true, false}));
}

// ── TPC-H-like shape ──

TEST(StringInMultiple, TpchQ12ShipmodeInSmall) {
    // Q12's `l_shipmode IN ("MAIL", "SHIP")` — small pattern set,
    // short strings. Validate using a mini-dataset.
    auto got = run_in_multiple(
        {"MAIL", "RAIL", "SHIP", "TRUCK", "AIR", "MAIL", "FOB", "SHIP"},
        {"MAIL", "SHIP"});
    EXPECT_EQ(got, (std::vector<bool>{
        true, false, true, false, false, true, false, true}));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
