#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>

#include "string/string.hpp"
#include "string/string_column.hpp"
#include "memory/column.hpp"

namespace gamax {
namespace test {

struct SortTestParams {
    string_impl_id impl;
    sort_order order;
    bool use_prefix_sort;
    
    friend std::ostream& operator<<(std::ostream& os, const SortTestParams& params) {
        os << "Impl=" << static_cast<int>(params.impl)
           << "_Order=" << (params.order == sort_order::ASCENDING ? "ASC" : "DESC")
           << "_PrefixSort=" << params.use_prefix_sort;
        return os;
    }
};

class StringSortTest : public ::testing::TestWithParam<SortTestParams> {
protected:
    void SetUp() override { 
        cudaStreamCreate(&stream_); 
    }
    
    void TearDown() override { 
        cudaStreamDestroy(stream_); 
    }

    cudaStream_t stream_{};

    static std::pair<std::vector<char>, std::vector<int32_t>>
    prepareFlatData(const std::vector<std::string>& strings) {
        std::vector<char> flat;
        std::vector<int32_t> lens;
        flat.reserve([&]{ 
            size_t n = 0; 
            for(auto const& s: strings) n += s.size(); 
            return n; 
        }());
        lens.reserve(strings.size());
        for (auto const& s : strings) {
            lens.push_back(static_cast<int32_t>(s.size()));
            flat.insert(flat.end(), s.begin(), s.end());
        }
        return {std::move(flat), std::move(lens)};
    }

    static std::vector<std::string> getTestDataset() {
        return {
            "zebra",
            "apple",
            "banana",
            "ZEBRA",
            "APPLE",
            "cherry",
            "date with spaces and more text",
            "apricot",
            "",
            "a",
            "aa",
            "aaa",
            "long string with many characters to test whatever bad things that might happen in sorting",
            "another long string that is slightly different and needs overflow",
            "medium length string",
            "short",
            "this is a test string with punctuation!!!",
            "123 numeric prefix",
            "456 another numeric prefix"
        };
    }

    static std::vector<size_type> computeExpectedIndices(
        const std::vector<std::string>& strings,
        sort_order order)
    {
        std::vector<size_type> indices(strings.size());
        for (size_type i = 0; i < strings.size(); ++i) {
            indices[i] = i;
        }
        
        if (order == sort_order::ASCENDING) {
            std::sort(indices.begin(), indices.end(), 
                [&strings](size_type a, size_type b) {
                    return strings[a] < strings[b];
                });
        } else {
            std::sort(indices.begin(), indices.end(), 
                [&strings](size_type a, size_type b) {
                    return strings[a] > strings[b];
                });
        }
        
        return indices;
    }
};

TEST_P(StringSortTest, ArgsortProducesCorrectIndices) {
    const auto params = GetParam();
    const auto test_strings = getTestDataset();
    
    gamax::string_op_options options{};
    options.use_prefix_sort = params.use_prefix_sort;

    auto [flat, lens] = prepareFlatData(test_strings);
    auto str_col_ptr = make_string_column(
        flat.data(),
        lens.data(),
        static_cast<size_type>(test_strings.size()),
        stream_,
        params.impl);

    ASSERT_NE(str_col_ptr, nullptr);
    auto& str_col = dynamic_cast<gamax::string_column&>(*str_col_ptr);
    auto str_view = str_col.view_s();

    auto result = argsort_strings(str_view, params.order, stream_, options);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->type().id(), gamax::type_id::INT32);
    EXPECT_EQ(result->size(), static_cast<size_type>(test_strings.size()));

    auto host_indices = gamax::copy_column_to_host<size_type>(result->view(), stream_);
    cudaStreamSynchronize(stream_);
    
    auto expected_indices = computeExpectedIndices(test_strings, params.order);
    
    for (size_type i = 0; i < result->size(); ++i) {
        EXPECT_EQ(host_indices[i], expected_indices[i]) 
            << "Mismatch at position " << i 
            << ": expected index " << expected_indices[i] 
            << " (string: '" << test_strings[expected_indices[i]] << "'), "
            << "got index " << host_indices[i] 
            << " (string: '" << test_strings[host_indices[i]] << "')";
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllSortVariants,
    StringSortTest,
    ::testing::Values(
        SortTestParams{string_impl_id::ARROW_STRING, sort_order::ASCENDING, false},
        SortTestParams{string_impl_id::ARROW_STRING, sort_order::ASCENDING, true},
        SortTestParams{string_impl_id::ARROW_STRING, sort_order::DESCENDING, false},
        SortTestParams{string_impl_id::ARROW_STRING, sort_order::DESCENDING, true}
    )
);

} // namespace test
} // namespace gamax
