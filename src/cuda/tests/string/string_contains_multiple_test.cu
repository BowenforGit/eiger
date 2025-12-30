#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <utility>

#include "string/string.hpp"
#include "string/string_column.hpp"
#include "memory/column.hpp"

namespace gamax {
namespace test {

enum class ContainsMultipleDatasetId { 
    Basic,           // Basic sequential pattern matching
    NoMatch,         // Patterns exist but not in order
    Overlap,         // Patterns would match if overlapping allowed
    Empty,           // Edge cases with empty strings/patterns
    SinglePattern    // Single pattern (should behave like contains)
};

struct ContainsMultipleTestParams {
    string_impl_id impl;
    ContainsMultipleDatasetId dataset_id;
    int access_width;

    friend std::ostream& operator<<(std::ostream& os, const ContainsMultipleTestParams& params) {
        os << "Impl=" << static_cast<int>(params.impl)
           << "_Dataset=" << static_cast<int>(params.dataset_id)
           << "_AccessWidth=" << params.access_width;
        return os;
    }
};

struct ContainsMultipleDataset {
    std::vector<std::string> strings;
    std::vector<std::string> patterns;
    std::vector<bool> expected;
};

class ContainsMultipleTest : public ::testing::TestWithParam<ContainsMultipleTestParams> {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }

    cudaStream_t stream_{};

    static std::pair<std::vector<char>, std::vector<int32_t>>
    prepareFlatData(const std::vector<std::string>& strings) {
        std::vector<char> flat;
        std::vector<int32_t> lens;
        flat.reserve([&]{ size_t n=0; for(auto const& s: strings) n+=s.size(); return n; }());
        lens.reserve(strings.size());
        for (auto const& s : strings) {
            lens.push_back(static_cast<int32_t>(s.size()));
            flat.insert(flat.end(), s.begin(), s.end());
        }
        return {std::move(flat), std::move(lens)};
    }

    static ContainsMultipleDataset getDataset(ContainsMultipleDatasetId id) {
        switch(id) {
            case ContainsMultipleDatasetId::Basic: {
                // Test basic sequential pattern matching
                return {
                    {
                        "The quick brown fox jumps over the lazy dog",  // all patterns in order
                        "quick fox lazy",                                // all patterns in order, minimal
                        "The slow brown fox jumps over the lazy dog",   // missing "quick"
                        "The quick brown fox",                          // missing "lazy"
                        "lazy quick fox",                               // wrong order
                        "The quick brown fox jumps over the lazy dog and more text after"  // match with extra
                    },
                    {"quick", "fox", "lazy"},
                    {true, true, false, false, false, true}
                };
            }
            case ContainsMultipleDatasetId::NoMatch: {
                // Patterns exist but not in required order
                return {
                    {
                        "CBA",           // reversed
                        "BAC",           // wrong order
                        "ACB",           // wrong order
                        "ABC",           // correct order
                        "AABBCC",        // correct order with duplicates
                        "A B C"          // correct order with spaces
                    },
                    {"A", "B", "C"},
                    {false, false, false, true, true, true}
                };
            }
            case ContainsMultipleDatasetId::Overlap: {
                // Test that overlapping is NOT allowed
                return {
                    {
                        "abcdef",        // "abc" at 0, "cde" needs to start at 3 (no overlap)
                        "abcde",         // "abc" at 0, "cde" at 2 would overlap - not allowed
                        "abcXcdef",      // "abc" at 0, "cde" at 4 (no overlap) - should match
                        "aaaa",          // "aa" at 0, next "aa" needs to start at 2 (valid)
                        "aaa"            // "aa" at 0, next "aa" at 1 would overlap - not allowed
                    },
                    {"abc", "cde"},
                    {true, false, true, false, false}  // Note: last two use different patterns below
                };
            }
            case ContainsMultipleDatasetId::Empty: {
                // Edge cases
                return {
                    {
                        "",              // empty string
                        "hello",         // non-empty string
                        "test string",   // normal string
                        "a"              // single char
                    },
                    {"test"},
                    {false, false, true, false}
                };
            }
            case ContainsMultipleDatasetId::SinglePattern: {
                // Single pattern should behave like contains
                return {
                    {
                        "hello world",
                        "world hello",
                        "no match here",
                        "worldworld",
                        ""
                    },
                    {"world"},
                    {true, true, false, true, false}
                };
            }
        }
        return {{}, {}, {}};
    }
};

TEST_P(ContainsMultipleTest, ContainsMultiplePatternsInOrder) {
    const auto params = GetParam();
    const auto ds = getDataset(params.dataset_id);
    
    gamax::string_op_options options{};
    options.access_width = params.access_width;

    // Prepare string column
    auto [flat, lens] = prepareFlatData(ds.strings);
    auto str_col_ptr = make_string_column(
        flat.data(),
        lens.data(),
        static_cast<size_type>(ds.strings.size()),
        stream_,
        params.impl);

    ASSERT_NE(str_col_ptr, nullptr);
    auto& str_col = dynamic_cast<gamax::string_column&>(*str_col_ptr);
    auto str_view = str_col.view_s();

    // Prepare pattern scalars
    std::vector<std::unique_ptr<string_scalar>> pattern_scalars;
    std::vector<string_scalar_view> pattern_views;
    pattern_scalars.reserve(ds.patterns.size());
    pattern_views.reserve(ds.patterns.size());
    
    for (const auto& p : ds.patterns) {
        pattern_scalars.push_back(make_string_scalar(p, stream_));
        pattern_views.push_back(pattern_scalars.back()->view());
    }

    // Run contains_multiple
    auto result = contains_multiple(str_view, pattern_views, stream_, options);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->type().id(), gamax::type_id::BOOL8);
    EXPECT_EQ(result->size(), static_cast<size_type>(ds.expected.size()));

    auto host = gamax::copy_column_to_host<bool>(result->view(), stream_);
    cudaStreamSynchronize(stream_);
    for (size_type i = 0; i < result->size(); ++i) {
        EXPECT_EQ(static_cast<bool>(host[i]), ds.expected[i]) 
            << "Mismatch at index " << i << " for string: \"" << ds.strings[i] << "\"";
    }
}

// Test with empty pattern vector (edge case - should return all true)
TEST_F(ContainsMultipleTest, EmptyPatternVector) {
    std::vector<std::string> strings = {"hello", "world", ""};
    auto [flat, lens] = prepareFlatData(strings);
    
    auto str_col_ptr = make_string_column(
        flat.data(),
        lens.data(),
        static_cast<size_type>(strings.size()),
        stream_,
        string_impl_id::ARROW_STRING);
    
    auto& str_col = dynamic_cast<gamax::string_column&>(*str_col_ptr);
    auto str_view = str_col.view_s();

    std::vector<string_scalar_view> empty_patterns;
    auto result = contains_multiple(str_view, empty_patterns, stream_);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), static_cast<size_type>(strings.size()));

    auto host = gamax::copy_column_to_host<bool>(result->view(), stream_);
    cudaStreamSynchronize(stream_);
    for (size_type i = 0; i < result->size(); ++i) {
        EXPECT_TRUE(static_cast<bool>(host[i])) << "Empty patterns should match all strings";
    }
}

// Test overlap behavior specifically
TEST_F(ContainsMultipleTest, OverlapNotAllowed) {
    // Test case: "aaaa" with patterns ["aa", "aa"]
    // First "aa" matches at position 0, second "aa" must start at position 2
    std::vector<std::string> strings = {
        "aaaa",   // "aa" at 0, "aa" at 2 - should match (no overlap)
        "aaa",    // "aa" at 0, next "aa" would need pos 2 but string ends - no match
        "aa aa",  // "aa" at 0, "aa" at 3 - should match
        "aaaaa"   // "aa" at 0, "aa" at 2 - should match
    };
    std::vector<bool> expected = {true, false, true, true};
    
    auto [flat, lens] = prepareFlatData(strings);
    auto str_col_ptr = make_string_column(
        flat.data(),
        lens.data(),
        static_cast<size_type>(strings.size()),
        stream_,
        string_impl_id::ARROW_STRING);
    
    auto& str_col = dynamic_cast<gamax::string_column&>(*str_col_ptr);
    auto str_view = str_col.view_s();

    auto p1 = make_string_scalar("aa", stream_);
    auto p2 = make_string_scalar("aa", stream_);
    std::vector<string_scalar_view> patterns = {p1->view(), p2->view()};

    auto result = contains_multiple(str_view, patterns, stream_);

    auto host = gamax::copy_column_to_host<bool>(result->view(), stream_);
    cudaStreamSynchronize(stream_);
    for (size_type i = 0; i < result->size(); ++i) {
        EXPECT_EQ(static_cast<bool>(host[i]), expected[i]) 
            << "Mismatch at index " << i << " for string: \"" << strings[i] << "\"";
    }
}

INSTANTIATE_TEST_SUITE_P(
    ContainsMultiple,
    ContainsMultipleTest,
    ::testing::Values(
        ContainsMultipleTestParams{string_impl_id::ARROW_STRING, ContainsMultipleDatasetId::Basic, 1},
        ContainsMultipleTestParams{string_impl_id::ARROW_STRING, ContainsMultipleDatasetId::Basic, 4},
        ContainsMultipleTestParams{string_impl_id::ARROW_STRING, ContainsMultipleDatasetId::Basic, 8},
        ContainsMultipleTestParams{string_impl_id::ARROW_STRING, ContainsMultipleDatasetId::NoMatch, 4},
        ContainsMultipleTestParams{string_impl_id::ARROW_STRING, ContainsMultipleDatasetId::Empty, 4},
        ContainsMultipleTestParams{string_impl_id::ARROW_STRING, ContainsMultipleDatasetId::SinglePattern, 4}
    ));

} // namespace test
} // namespace gamax
