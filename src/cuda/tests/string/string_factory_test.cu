#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

#include "string/string_column.hpp"
#include "string/string_impl.cuh"
#include "memory/column.hpp"

namespace gamax {
namespace test {

// Parameterized test: runs for each (impl, dataset) pair
enum class DatasetId { Mixed, Short, Long };

struct TestParams {
    string_impl_id impl;
    DatasetId dataset_id;
};

class StringFactoryTest : public ::testing::TestWithParam<TestParams> {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
    }

    void TearDown() override {
        cudaStreamDestroy(stream_);
    }

    cudaStream_t stream_;

    // Helper function to create flat data and lengths from vector of strings
    std::pair<std::vector<char>, std::vector<int32_t>> 
    prepareFlatData(const std::vector<std::string>& strings) {
        std::vector<char> flat_data;
        std::vector<int32_t> lengths;
        
        for (const auto& str : strings) {
            lengths.push_back(static_cast<int32_t>(str.size()));
            flat_data.insert(flat_data.end(), str.begin(), str.end());
        }
        
        return {flat_data, lengths};
    }

    // Centralized test datasets to avoid duplication
    static const std::vector<std::string>& datasetMixed() {
        static const std::vector<std::string> v = {
            "hi",
            "test",
            "hello world, this is a cool string layout, don't you think?",
            "abcd",
            "this is a much longer string that definitely needs overflow storage"
        };
        return v;
    }

    static const std::vector<std::string>& datasetShort() {
        static const std::vector<std::string> v = {
            "",
            "a",
            "hello",
            "test string",
            "this"
        };
        return v;
    }

    static const std::vector<std::string>& datasetLong() {
        static const std::vector<std::string> v = {
            "hi, my name is Bob, what's your name?",
            "test testing 1 2 3 4 5 6 7 8 9 10",
            "hello world, this is a cool string layout, don't you think?",
            "abcdefghijklmnopqrstuvwxyz",
            "this is a much longer string that definitely needs overflow storage"
        };
        return v;
    }

    static const std::vector<std::string>& getDataset(DatasetId id) {
        switch (id) {
            case DatasetId::Mixed: return datasetMixed();
            case DatasetId::Short: return datasetShort();
            case DatasetId::Long: return datasetLong();
        }
        return datasetMixed();
    }

    // Helper to verify string content by reading back from device
    void verifyStringContent(
        const string_column& col,
        const std::vector<std::string>& expected_strings)
    {
        using german_impl_t = gamax::detail::id_to_string_impl<string_impl_id::GERMAN_STRING>::string_col_impl;
        using arrow_impl_t = gamax::detail::id_to_string_impl<string_impl_id::ARROW_STRING>::string_col_impl;
        
        ASSERT_EQ(col.size(), expected_strings.size());
        
        if (col.impl() == string_impl_id::GERMAN_STRING) {
            verifyGermanStringLayout(col, expected_strings);
        } else if (col.impl() == string_impl_id::ARROW_STRING) {
            verifyArrowStringLayout(col, expected_strings);
        }
    }

    void verifyGermanStringLayout(
        const string_column& col,
        const std::vector<std::string>& expected_strings)
    {
        using german_impl_t = gamax::detail::id_to_string_impl<string_impl_id::GERMAN_STRING>::string_col_impl;
        using header_entry_t = typename german_impl_t::header_entry_type;
        
        constexpr size_t inline_limit = german_impl_t::max_inlined_len(); // prefix + buffer_offset union
        constexpr size_t prefix_len = german_impl_t::prefix_len();
        
        // Copy metadata (headers) from device to host
        std::vector<header_entry_t> headers(col.size());
        cudaMemcpy(
            headers.data(),
            col.view_s().meta(),
            col.size() * sizeof(header_entry_t),
            cudaMemcpyDeviceToHost);
        
        // Copy data buffer from device to host
        std::vector<char> data_buffer(col.bytes());
        cudaMemcpy(
            data_buffer.data(),
            col.buffer().data(), // somehow col.view_s().data() gives an error
            col.bytes(),
            cudaMemcpyDeviceToHost);
        
        // Verify each string
        for (size_t i = 0; i < expected_strings.size(); ++i) {
            const auto& expected = expected_strings[i];
            const auto& header = headers[i];
            
            // Verify length is stored correctly
            EXPECT_EQ(header.length(), expected.size()) 
                << "String " << i << " has incorrect length";
            
            // Reconstruct the string from header and buffer
            std::string reconstructed;
            
            if (expected.size() <= inline_limit) {
                // String should be fully inlined
                for (size_t j = 0; j < expected.size(); ++j) {
                    reconstructed.push_back(header.charAt(j));
                }
            } else {
                // String has prefix + overflow in buffer
                // First prefix_len bytes are in header
                for (size_t j = 0; j < prefix_len; ++j) {
                    reconstructed.push_back(header.charAt(j));
                }
                // Rest is in overflow buffer
                size_t overflow_offset = header.offset();
                size_t overflow_len = expected.size() - prefix_len;
                for (size_t j = 0; j < overflow_len; ++j) {
                    reconstructed.push_back(data_buffer[overflow_offset + j]);
                }
            }
            
            EXPECT_EQ(reconstructed, expected)
                << "String " << i << " content mismatch. Expected: '" 
                << expected << "', Got: '" << reconstructed << "'";
        }
    }

    void verifyArrowStringLayout(
        const string_column& col,
        const std::vector<std::string>& expected_strings)
    {
        using arrow_impl_t = gamax::detail::id_to_string_impl<string_impl_id::ARROW_STRING>::string_col_impl;
        using offset_t = gamax::size_type;
        
        // Copy offsets from device to host (num_strings + 1 offsets)
        std::vector<offset_t> offsets(col.size() + 1);
        cudaMemcpy(
            offsets.data(),
            col.meta_buffer().data(),
            (col.size() + 1) * sizeof(offset_t),
            cudaMemcpyDeviceToHost);
        
        // Copy data buffer from device to host
        std::vector<char> data_buffer(col.bytes());
        cudaMemcpy(
            data_buffer.data(),
            col.buffer().data(),
            col.bytes(),
            cudaMemcpyDeviceToHost);
        
        // Verify each string
        for (size_t i = 0; i < expected_strings.size(); ++i) {
            const auto& expected = expected_strings[i];
            
            offset_t start = offsets[i];
            offset_t end = offsets[i + 1];
            offset_t length = end - start;
            
            EXPECT_EQ(length, expected.size())
                << "String " << i << " has incorrect length";
            
            std::string reconstructed(data_buffer.begin() + start, 
                                     data_buffer.begin() + end);
            
            EXPECT_EQ(reconstructed, expected)
                << "String " << i << " content mismatch. Expected: '" 
                << expected << "', Got: '" << reconstructed << "'";
        }
        
        // Verify first offset is 0
        EXPECT_EQ(offsets[0], 0) << "First offset should be 0";
    }

    void testStringLayout(
        string_impl_id impl,
        const std::vector<std::string>& test_strings)
    {
        auto [flat_data, lengths] = prepareFlatData(test_strings);
        
        // Create string column
        auto col = make_string_column(
            flat_data.data(),
            lengths.data(),
            test_strings.size(),
            stream_,
            impl);
        
        cudaStreamSynchronize(stream_);
        
        ASSERT_NE(col, nullptr);
        auto& col_ref = dynamic_cast<gamax::string_column&>(*col);
        EXPECT_EQ(col_ref.impl(), impl);
        EXPECT_EQ(col_ref.size(), test_strings.size());
        
        if (impl == string_impl_id::GERMAN_STRING) {
            verifyGermanStringLayout(col_ref, test_strings);
        } else if (impl == string_impl_id::ARROW_STRING) {
            verifyArrowStringLayout(col_ref, test_strings);
        }
    }
};

TEST_P(StringFactoryTest, StringFactoryLayouts) {
    const auto params = GetParam();
    const auto& test_strings = getDataset(params.dataset_id);
    testStringLayout(params.impl, test_strings);
}

// Instantiate combinations: both layouts x all datasets
INSTANTIATE_TEST_SUITE_P(
    AllLayoutsAndDatasets,
    StringFactoryTest,
    ::testing::Values(
        TestParams{string_impl_id::GERMAN_STRING, DatasetId::Mixed},
        TestParams{string_impl_id::ARROW_STRING, DatasetId::Mixed},
        TestParams{string_impl_id::GERMAN_STRING, DatasetId::Short},
        TestParams{string_impl_id::ARROW_STRING, DatasetId::Short},
        TestParams{string_impl_id::GERMAN_STRING, DatasetId::Long},
        TestParams{string_impl_id::ARROW_STRING, DatasetId::Long}
    )
);

} // namespace test
} // namespace gamax
