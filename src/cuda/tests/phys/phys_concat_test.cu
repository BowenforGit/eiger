// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Unit tests for `phys::concat` — vertical (UNION ALL)
// concatenation of N table_views. Numeric columns get a per-input
// memcpy into a pre-sized output; STRING columns get the byte buffer
// concatenated with the offsets shifted per input.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "phys/phys.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"

namespace phys = gamax::phys;
using gamax::data_type;
using gamax::type_id;

namespace {

data_type t(type_id id) { return data_type{id}; }
phys::Context default_ctx() { return phys::Context{cudaStreamDefault}; }

template <typename T>
gamax::column_ptr mk_col(const std::vector<T>& v) {
    return gamax::make_column_from_host(const_cast<T*>(v.data()), static_cast<gamax::size_type>(v.size()), cudaStreamDefault);
}

gamax::column_ptr mk_string_col(const std::vector<std::string>& ss) {
    std::vector<char> flat;
    std::vector<int32_t> lens;
    for (const auto& s : ss) {
        lens.push_back(static_cast<int32_t>(s.size()));
        flat.insert(flat.end(), s.begin(), s.end());
    }
    auto ptr = gamax::make_string_column(
        flat.data(), lens.data(),
        static_cast<gamax::size_type>(ss.size()),
        cudaStreamDefault, gamax::ARROW_STRING);
    return std::unique_ptr<gamax::column>(std::move(ptr));
}

// Read a STRING column back through the non-owning view metadata.
std::vector<std::string> dump_string_col(
    gamax::column_view col,
    cudaStream_t stream = cudaStreamDefault)
{
    const auto n = col.size();
    std::vector<gamax::size_type> offsets(n + 1);
    cudaMemcpy(offsets.data(), col.auxiliary_data(),
               (n + 1) * sizeof(gamax::size_type),
               cudaMemcpyDeviceToHost);
    const auto total = offsets.back();
    std::string bytes(total, '\0');
    if (total > 0) {
        cudaMemcpy(bytes.data(), col.data<void>(), total,
                   cudaMemcpyDeviceToHost);
    }
    std::vector<std::string> out;
    out.reserve(n);
    for (gamax::size_type i = 0; i < n; ++i) {
        out.emplace_back(bytes.data() + offsets[i],
                         offsets[i + 1] - offsets[i]);
    }
    return out;
}

} // namespace

TEST(PhysConcat, TwoNumericTables) {
    std::vector<int32_t> a = {1, 2, 3};
    std::vector<int32_t> b = {10, 20};
    std::vector<gamax::column_ptr> ca;
    ca.push_back(mk_col(a));
    auto ta = gamax::make_table(std::move(ca));
    std::vector<gamax::column_ptr> cb;
    cb.push_back(mk_col(b));
    auto tb = gamax::make_table(std::move(cb));

    auto out = phys::concat({ta->view(), tb->view()}, default_ctx());
    ASSERT_EQ(out->num_rows(), 5);
    ASSERT_EQ(out->num_columns(), 1);
    auto host = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    std::vector<int32_t> expected = {1, 2, 3, 10, 20};
    for (int i = 0; i < 5; ++i) EXPECT_EQ(host[i], expected[i]);
}

TEST(PhysConcat, MultiColumnNumericPreservesOrder) {
    // Two tables with (INT32, FLOAT64). After concat, rows from the
    // first input precede those from the second; per-row column
    // alignment is preserved.
    std::vector<int32_t> a_i = {1, 2};
    std::vector<double>  a_f = {1.5, 2.5};
    std::vector<int32_t> b_i = {30, 40, 50};
    std::vector<double>  b_f = {3.5, 4.5, 5.5};
    std::vector<gamax::column_ptr> ca;
    ca.push_back(mk_col(a_i));
    ca.push_back(mk_col(a_f));
    auto ta = gamax::make_table(std::move(ca));
    std::vector<gamax::column_ptr> cb;
    cb.push_back(mk_col(b_i));
    cb.push_back(mk_col(b_f));
    auto tb = gamax::make_table(std::move(cb));

    auto out = phys::concat({ta->view(), tb->view()}, default_ctx());
    ASSERT_EQ(out->num_rows(), 5);
    ASSERT_EQ(out->num_columns(), 2);
    auto h_i = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    auto h_f = gamax::copy_column_to_host<double>(out->view()[1], cudaStreamDefault);
    std::vector<int32_t> expected_i = {1, 2, 30, 40, 50};
    std::vector<double>  expected_f = {1.5, 2.5, 3.5, 4.5, 5.5};
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(h_i[i], expected_i[i]);
        EXPECT_DOUBLE_EQ(h_f[i], expected_f[i]);
    }
}

TEST(PhysConcat, StringColumnsOffsetsShifted) {
    // Strings of varied length exercise the offset-shift path.
    std::vector<std::string> a = {"alpha", "", "beta"};
    std::vector<std::string> b = {"gamma", "delta"};
    std::vector<gamax::column_ptr> ca;
    ca.push_back(mk_string_col(a));
    auto ta = gamax::make_table(std::move(ca));
    std::vector<gamax::column_ptr> cb;
    cb.push_back(mk_string_col(b));
    auto tb = gamax::make_table(std::move(cb));

    auto out = phys::concat({ta->view(), tb->view()}, default_ctx());
    ASSERT_EQ(out->num_rows(), 5);
    auto host = dump_string_col(out->view()[0]);
    std::vector<std::string> expected = {
        "alpha", "", "beta", "gamma", "delta"};
    EXPECT_EQ(host, expected);
}

TEST(PhysConcat, MixedNumericAndString) {
    // Schema: (STRING, INT64). Two inputs, out shape 2+3 = 5.
    std::vector<std::string> a_s = {"x", "yy"};
    std::vector<int64_t> a_i = {1, 2};
    std::vector<std::string> b_s = {"zzz", "aa", "bbbb"};
    std::vector<int64_t> b_i = {10, 20, 30};
    std::vector<gamax::column_ptr> ca;
    ca.push_back(mk_string_col(a_s));
    ca.push_back(mk_col(a_i));
    auto ta = gamax::make_table(std::move(ca));
    std::vector<gamax::column_ptr> cb;
    cb.push_back(mk_string_col(b_s));
    cb.push_back(mk_col(b_i));
    auto tb = gamax::make_table(std::move(cb));

    auto out = phys::concat({ta->view(), tb->view()}, default_ctx());
    ASSERT_EQ(out->num_rows(), 5);
    ASSERT_EQ(out->num_columns(), 2);
    auto h_s = dump_string_col(out->view()[0]);
    auto h_i = gamax::copy_column_to_host<int64_t>(out->view()[1], cudaStreamDefault);
    std::vector<std::string> expected_s = {"x", "yy", "zzz", "aa", "bbbb"};
    std::vector<int64_t>     expected_i = {1, 2, 10, 20, 30};
    EXPECT_EQ(h_s, expected_s);
    for (int i = 0; i < 5; ++i) EXPECT_EQ(h_i[i], expected_i[i]);
}

TEST(PhysConcat, ThreeWayNary) {
    // Pin the N-ary path: allocate sum(sizes) in one go, no binary
    // folding.
    std::vector<int32_t> a = {1};
    std::vector<int32_t> b = {2, 3};
    std::vector<int32_t> c = {4, 5, 6};
    std::vector<gamax::column_ptr> ca;  ca.push_back(mk_col(a));
    std::vector<gamax::column_ptr> cb;  cb.push_back(mk_col(b));
    std::vector<gamax::column_ptr> cc;  cc.push_back(mk_col(c));
    auto ta = gamax::make_table(std::move(ca));
    auto tb = gamax::make_table(std::move(cb));
    auto tc = gamax::make_table(std::move(cc));

    auto out = phys::concat(
        {ta->view(), tb->view(), tc->view()}, default_ctx());
    ASSERT_EQ(out->num_rows(), 6);
    auto host = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    std::vector<int32_t> expected = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; ++i) EXPECT_EQ(host[i], expected[i]);
}

TEST(PhysConcat, EmptyInputAmongOthersIsNoOp) {
    // A 0-row input in the middle should be tolerated: output skips
    // it and preserves the non-empty ones' row order.
    std::vector<int32_t> a = {1, 2};
    std::vector<int32_t> empty;
    std::vector<int32_t> c = {3};
    std::vector<gamax::column_ptr> ca;  ca.push_back(mk_col(a));
    std::vector<gamax::column_ptr> cb;  cb.push_back(mk_col(empty));
    std::vector<gamax::column_ptr> cc;  cc.push_back(mk_col(c));
    auto ta = gamax::make_table(std::move(ca));
    auto tb = gamax::make_table(std::move(cb));
    auto tc = gamax::make_table(std::move(cc));

    auto out = phys::concat(
        {ta->view(), tb->view(), tc->view()}, default_ctx());
    ASSERT_EQ(out->num_rows(), 3);
    auto host = gamax::copy_column_to_host<int32_t>(out->view()[0], cudaStreamDefault);
    std::vector<int32_t> expected = {1, 2, 3};
    for (int i = 0; i < 3; ++i) EXPECT_EQ(host[i], expected[i]);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
