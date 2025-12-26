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

enum class Operation { Equal, StartsWith, EndsWith, Contains };
enum class OpDatasetId { Case1, Case2 };

struct TestParams {
    string_impl_id impl;
    Operation op;
    OpDatasetId dataset_id;
    int cg_size;
    int access_width;
    bool use_kmp;

    friend std::ostream& operator<<(std::ostream& os, const TestParams& params) {
        os << "Impl=" << static_cast<int>(params.impl)
           << "_Op=" << static_cast<int>(params.op)
           << "_Dataset=" << static_cast<int>(params.dataset_id)
           << "_CGSize=" << params.cg_size
           << "_AccessWidth=" << params.access_width
           << "_UseKMP=" << params.use_kmp;
        return os;
    }
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
            case Operation::Equal: {
                if (id == OpDatasetId::Case1) {
                    return {
                        {
                            "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                            "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                            "ABCD EFGH IJKL MNOP QRST",
                            "",
                            "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456 7890 !@#$%^&*()",
                            "ABCD EFGH IJKL MNOP !@#$ UVWX YZ12 3456 7890"
                        },
                        "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                        {true,true,false,false,false,false}
                    };
                } else {
                    return {
                        {"test","test","testing","string",""},
                        "test",
                        {true,true,false,false,false}
                    };
                }
            }
            case Operation::StartsWith: {
                if (id == OpDatasetId::Case1) {
                    return {
                        {
                            "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456 7890",
                            "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456 abcde",
                            "ABCD EFGH IJKL MNOP QRST",
                            "",
                            "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456 7890 !@#$%^&*()",
                            "ABCD EFGH IJKL MNOP !@#$ UVWX YZ12 3456 7890"
                        },
                        "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                        {true,true,false,false,true,false}
                    };
                } else {
                    return {
                        {
                            "hello world, my name is Eiger.",
                            "world world world",
                            "he",
                            "hell hello hallo guten morgen",
                            "hello world!!!"
                        },
                        "hello",
                        {true,false,false,false,true}
                    };
                }
            }
            case Operation::EndsWith: {
                if (id == OpDatasetId::Case1) {
                    return {
                        {
                            "7890 ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                            "abcde ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                            "ABCD EFGH IJKL MNOP QRST",
                            "",
                            "7890 !@#$%^&*() ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                            "ABCD EFGH IJKL MNOP !@#$ UVWX YZ12 3456"
                        },
                        "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                        {true,true,false,false,true,false}
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
                        {
                            "fa09 901jd ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456 7890 djailsja",
                            "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456 abcde",
                            "ABCD EFGH IJKL MNOP QRST",
                            "",
                            "jflak;jfslajla; ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456 7890 !@#$%^&*()",
                            "ABCD EFGH IJKL MNOP !@#$ UVWX YZ12 3456 7890"
                        },
                        "ABCD EFGH IJKL MNOP QRST UVWX YZ12 3456",
                        {true,true,false,false,true,false}
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
        cudaStream_t stream,
        string_op_options options)
    {
        switch(op) {
            case Operation::Equal:      return equal(in, param, stream, options);
            case Operation::StartsWith: return startswith(in, param, stream, options);
            case Operation::EndsWith:   return endswith(in, param, stream, options);
            case Operation::Contains:   return contains(in, param, stream, options);
        }
        return nullptr;
    }
};

TEST_P(StringOpsTest, StringOperationProducesExpectedMask) {
    const auto params = GetParam();
    const auto ds = getDataset(params.op, params.dataset_id);
    
    gamax::string_op_options options{};
    options.cg_size = params.cg_size;
    options.access_width = params.access_width;
    options.use_kmp = params.use_kmp;

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
    auto result = runOp(params.op, str_view, scalar_ptr->view(), stream_, options);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->type().id(), gamax::type_id::BOOL8);
    EXPECT_EQ(result->size(), static_cast<size_type>(ds.expected.size()));

    auto host = gamax::copy_column_to_host<bool>(result->view(), stream_);
    cudaStreamSynchronize(stream_);
    for (size_type i = 0; i < result->size(); ++i) {
        EXPECT_EQ(static_cast<bool>(host[i]), ds.expected[i]) << "Mismatch at index " << i;
    }
}

#define TEST_VALUES(IMPL, OP, USE_KMP) \
    TestParams{string_impl_id::IMPL, Operation::OP, OpDatasetId::Case1, 1, 1, USE_KMP}, \
    TestParams{string_impl_id::IMPL, Operation::OP, OpDatasetId::Case1, 1, 2, USE_KMP}, \
    TestParams{string_impl_id::IMPL, Operation::OP, OpDatasetId::Case1, 1, 4, USE_KMP}, \
    TestParams{string_impl_id::IMPL, Operation::OP, OpDatasetId::Case1, 1, 8, USE_KMP}, \
    TestParams{string_impl_id::IMPL, Operation::OP, OpDatasetId::Case1, 4, 1, USE_KMP}, \
    TestParams{string_impl_id::IMPL, Operation::OP, OpDatasetId::Case1, 4, 2, USE_KMP}, \
    TestParams{string_impl_id::IMPL, Operation::OP, OpDatasetId::Case1, 4, 4, USE_KMP}, \
    TestParams{string_impl_id::IMPL, Operation::OP, OpDatasetId::Case1, 4, 8, USE_KMP}

INSTANTIATE_TEST_SUITE_P(
    Equal,
    StringOpsTest,
    ::testing::Values(
        TEST_VALUES(ARROW_STRING,  Equal, false)
    ));

INSTANTIATE_TEST_SUITE_P(
    StartsWith,
    StringOpsTest,
    ::testing::Values(
        TEST_VALUES(ARROW_STRING,  StartsWith, false)
    ));

INSTANTIATE_TEST_SUITE_P(
    EndsWith,
    StringOpsTest,
    ::testing::Values(
        TEST_VALUES(ARROW_STRING,  EndsWith, false)
    ));

INSTANTIATE_TEST_SUITE_P(
    Contains,
    StringOpsTest,
    ::testing::Values(
        TEST_VALUES(ARROW_STRING,  Contains, false),
        TEST_VALUES(ARROW_STRING,  Contains, true)
    ));

} // namespace test
} // namespace gamax
