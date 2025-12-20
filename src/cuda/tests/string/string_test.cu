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

enum class Operation { StartsWith, EndsWith, Contains };
enum class OpDatasetId { Case1, Case2 };

struct TestParams {
    string_impl_id impl;
    Operation op;
    OpDatasetId dataset_id;
};

struct OpDataset {
    std::vector<std::string> strings;
    std::string query;
    std::vector<bool> expected;
};

class StringOpsTest : public ::testing::TestWithParam<TestParams> {
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

    static OpDataset getDataset(Operation op, OpDatasetId id) {
        switch(op) {
            case Operation::StartsWith: {
                if (id == OpDatasetId::Case1) {
                    return {
                        {"apple","app","banana","","application","apjdklajlskadsa"},
                        "app",
                        {true,true,false,false,true,false}
                    };
                } else {
                    return {
                        {"hello world, my name is Eiger.","world world world","he","hell hello hallo guten morgen","hello world!!!"},
                        "hello",
                        {true,false,false,false,true}
                    };
                }
            }
            case Operation::EndsWith: {
                if (id == OpDatasetId::Case1) {
                    return {
                        {"filejfklafsjkla.txt","docjdkflsjfl;ak.txt","imagekdjalkdkslajs;ld.png","","text"},
                        ".txt",
                        {true,true,false,false,false}
                    };
                } else {
                    return {
                        {"testing","ing","long","string",""},
                        "ing",
                        {true,true,false,true,false}
                    };
                }
            }
            case Operation::Contains: {
                if (id == OpDatasetId::Case1) {
                    return {
                        {"cuda is fast, looooool","fast","very fast indeed","slow",""},
                        "fast",
                        {true,true,true,false,false}
                    };
                } else {
                    return {
                        {"abc","bcd","cde","","abcdef"},
                        "cd",
                        {false,true,true,false,true}
                    };
                }
            }
        }
        // Fallback, should not reach
        return {{},"",{}};
    }

    static column_ptr runOp(
        Operation op,
        const gamax::string_column_view& in,
        const gamax::string_scalar_view& param,
        cudaStream_t stream)
    {
        switch(op) {
            case Operation::StartsWith: return startswith(in, param, stream);
            case Operation::EndsWith:   return endswith(in, param, stream);
            case Operation::Contains:   return contains(in, param, stream);
        }
        return nullptr;
    }
};

TEST_P(StringOpsTest, StringOperationProducesExpectedMask) {
    const auto params = GetParam();
    const auto ds = getDataset(params.op, params.dataset_id);

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

    auto scalar_ptr = make_string_scalar(ds.query, stream_);
    ASSERT_NE(scalar_ptr, nullptr);
    auto result = runOp(params.op, str_view, scalar_ptr->view(), stream_);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->type().id(), gamax::type_id::BOOL8);
    EXPECT_EQ(result->size(), static_cast<size_type>(ds.expected.size()));

    auto host = gamax::copy_column_to_host<bool>(result->view(), stream_);
    cudaStreamSynchronize(stream_);
    for (size_type i = 0; i < result->size(); ++i) {
        EXPECT_EQ(static_cast<bool>(host[i]), ds.expected[i]) << "Mismatch at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    StartsWithBothLayouts,
    StringOpsTest,
    ::testing::Values(
        TestParams{string_impl_id::GERMAN_STRING, Operation::StartsWith, OpDatasetId::Case1},
        TestParams{string_impl_id::ARROW_STRING,  Operation::StartsWith, OpDatasetId::Case1},
        TestParams{string_impl_id::GERMAN_STRING, Operation::StartsWith, OpDatasetId::Case2},
        TestParams{string_impl_id::ARROW_STRING,  Operation::StartsWith, OpDatasetId::Case2}
    ));

INSTANTIATE_TEST_SUITE_P(
    EndsWithBothLayouts,
    StringOpsTest,
    ::testing::Values(
        TestParams{string_impl_id::GERMAN_STRING, Operation::EndsWith, OpDatasetId::Case1},
        TestParams{string_impl_id::ARROW_STRING,  Operation::EndsWith, OpDatasetId::Case1},
        TestParams{string_impl_id::GERMAN_STRING, Operation::EndsWith, OpDatasetId::Case2},
        TestParams{string_impl_id::ARROW_STRING,  Operation::EndsWith, OpDatasetId::Case2}
    ));

INSTANTIATE_TEST_SUITE_P(
    ContainsBothLayouts,
    StringOpsTest,
    ::testing::Values(
        TestParams{string_impl_id::GERMAN_STRING, Operation::Contains, OpDatasetId::Case1},
        TestParams{string_impl_id::ARROW_STRING,  Operation::Contains, OpDatasetId::Case1},
        TestParams{string_impl_id::GERMAN_STRING, Operation::Contains, OpDatasetId::Case2},
        TestParams{string_impl_id::ARROW_STRING,  Operation::Contains, OpDatasetId::Case2}
    ));

} // namespace test
} // namespace gamax
