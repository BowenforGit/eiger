// Unit tests for `gamax::substring` — SQL `SUBSTRING(str FROM offset
// FOR length)`. `sql_offset` is 1-based (SQL convention). The
// implementation clamps `length` against remaining row width and
// returns empty when the start is past the row's end.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "memory/column.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"

namespace {

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
        flat.data(), lens.data(),
        static_cast<gamax::size_type>(strs.size()),
        stream);
}

// Read a string column back to host as a vector<std::string>, using
// the Arrow offsets layout directly. Arrow string columns store
// offsets as `gamax::size_type` (int64_t) — see `get_arrow_string_offsets`
// in string_factory.cu which forces the prefix_sum output to INT64.
std::vector<std::string> dump_string_column(
    const gamax::string_column& col,
    cudaStream_t stream = cudaStreamDefault)
{
    const auto n = col.size();
    std::vector<gamax::size_type> host_offsets(n + 1);
    cudaMemcpyAsync(
        host_offsets.data(),
        col.meta_buffer().data(),
        (n + 1) * sizeof(gamax::size_type),
        cudaMemcpyDeviceToHost,
        stream);
    cudaStreamSynchronize(stream);
    const gamax::size_type total_bytes = host_offsets.back();
    std::string host_data(total_bytes, '\0');
    if (total_bytes > 0) {
        cudaMemcpyAsync(
            host_data.data(),
            col.buffer().data(),
            total_bytes,
            cudaMemcpyDeviceToHost,
            stream);
        cudaStreamSynchronize(stream);
    }

    std::vector<std::string> out;
    out.reserve(n);
    for (gamax::size_type i = 0; i < n; ++i) {
        out.emplace_back(
            host_data.data() + host_offsets[i],
            host_offsets[i + 1] - host_offsets[i]);
    }
    return out;
}

std::vector<std::string> run_substring(
    const std::vector<std::string>& rows,
    gamax::size_type offset,
    gamax::size_type length,
    cudaStream_t stream = cudaStreamDefault)
{
    auto col = make_string_column_from_host(rows, stream);
    auto out = gamax::substring(col->view_s(), offset, length, stream);
    return dump_string_column(*out, stream);
}

}  // namespace

TEST(SubstringTest, FirstTwoChars) {
    // The TPC-H Q22 shape: take the first 2 chars.
    std::vector<std::string> rows = {
        "13-918-335-1736",
        "31-358-552-1736",
        "23-491-552-1736",
    };
    auto got = run_substring(rows, 1, 2);
    ASSERT_EQ(got.size(), rows.size());
    EXPECT_EQ(got[0], "13");
    EXPECT_EQ(got[1], "31");
    EXPECT_EQ(got[2], "23");
}

TEST(SubstringTest, MiddleSlice) {
    std::vector<std::string> rows = {"abcdef", "ABCDEFG", "xyz"};
    auto got = run_substring(rows, 2, 3);  // chars 2..4
    EXPECT_EQ(got[0], "bcd");
    EXPECT_EQ(got[1], "BCD");
    EXPECT_EQ(got[2], "yz");  // shorter-than-length rows clamp
}

TEST(SubstringTest, PastEndIsEmpty) {
    std::vector<std::string> rows = {"abc", "ab", "a"};
    auto got = run_substring(rows, 5, 2);  // start is past all rows
    EXPECT_EQ(got[0], "");
    EXPECT_EQ(got[1], "");
    EXPECT_EQ(got[2], "");
}

TEST(SubstringTest, ZeroLengthAllRowsEmpty) {
    std::vector<std::string> rows = {"abc", "de", "fgh"};
    auto got = run_substring(rows, 1, 0);
    EXPECT_EQ(got[0], "");
    EXPECT_EQ(got[1], "");
    EXPECT_EQ(got[2], "");
}

TEST(SubstringTest, VaryingLengths) {
    std::vector<std::string> rows = {
        "short", "medium_string", "a_much_longer_string_here", ""};
    auto got = run_substring(rows, 3, 5);
    EXPECT_EQ(got[0], "ort");        // only 3 chars available from offset 3
    EXPECT_EQ(got[1], "dium_");       // exactly 5 chars
    EXPECT_EQ(got[2], "much_");       // exactly 5 chars
    EXPECT_EQ(got[3], "");            // empty input → empty output
}

TEST(SubstringTest, OffsetZeroClampsToOne) {
    // The impl clamps non-positive offsets to 1 (SQL is
    // implementation-defined; TPC-H never goes below 1 so we simply
    // preserve the 1-based convention).
    std::vector<std::string> rows = {"abcdef"};
    auto got = run_substring(rows, 0, 3);
    EXPECT_EQ(got[0], "abc");
}

TEST(SubstringTest, EmptyColumn) {
    auto got = run_substring({}, 1, 2);
    EXPECT_TRUE(got.empty());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
