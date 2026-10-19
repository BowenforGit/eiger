// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Unit tests for `gamax::join::anti_join` — LEFT ANTI JOIN.
// Tests use tiny, hand-verified fixtures so correctness of the
// membership-mask-plus-negate path is easy to check.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/anti_join.hpp"
#include "operators/join/join.hpp"

namespace {

gamax::column_ptr make_int32_col(const std::vector<int32_t>& vs)
{
    return gamax::make_column_from_host(
        const_cast<int32_t*>(vs.data()),
        static_cast<gamax::size_type>(vs.size()),
        cudaStreamDefault);
}

std::unique_ptr<gamax::table> mk_table(std::vector<gamax::column_ptr>&& cols)
{
    return gamax::make_table(std::move(cols));
}

gamax::join::join_hint hash_hint()
{
    gamax::join::join_hint h;
    h.join_algo = gamax::join::HASH;
    h.mat_algo  = gamax::join::GFTR;
    return h;
}

std::vector<int32_t> to_host(gamax::column_view col)
{
    auto h = gamax::copy_column_to_host<int32_t>(col, cudaStreamDefault);
    std::vector<int32_t> out(col.size());
    for (gamax::size_type i = 0; i < col.size(); ++i) out[i] = h[i];
    return out;
}

} // namespace

// Left rows with keys not in right should survive; matching rows drop.
//
//   L (key, v): (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)
//   R (key):     2, 4, 6, 8
//   expected: keys {1, 3, 5} with their values, 3 rows
TEST(AntiJoin, BasicKeysNotInRight) {
    std::vector<gamax::column_ptr> lc;
    lc.push_back(make_int32_col({1, 2, 3, 4, 5}));
    lc.push_back(make_int32_col({10, 20, 30, 40, 50}));
    auto L = mk_table(std::move(lc));

    std::vector<gamax::column_ptr> rc;
    rc.push_back(make_int32_col({2, 4, 6, 8}));
    auto R = mk_table(std::move(rc));

    auto out = gamax::join::anti_join(L->view(), R->view(), hash_hint(), cudaStreamDefault);

    ASSERT_EQ(out->num_rows(), 3);
    ASSERT_EQ(out->num_columns(), 2);

    auto keys = to_host(out->get_column(0).view());
    auto vals = to_host(out->get_column(1).view());
    // Order is not guaranteed by the mask-compact path, so compare as sets.
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys, (std::vector<int32_t>{1, 3, 5}));
    // Values line up with surviving keys: key→value invariant k*10.
    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_EQ(vals[i], keys[i] * 10);
    }
}

// When every left key is in right, the anti-join produces no rows.
TEST(AntiJoin, AllMatchProducesEmpty) {
    std::vector<gamax::column_ptr> lc;
    lc.push_back(make_int32_col({1, 2, 3}));
    lc.push_back(make_int32_col({100, 200, 300}));
    auto L = mk_table(std::move(lc));

    std::vector<gamax::column_ptr> rc;
    rc.push_back(make_int32_col({1, 2, 3, 5, 7}));
    auto R = mk_table(std::move(rc));

    auto out = gamax::join::anti_join(L->view(), R->view(), hash_hint(), cudaStreamDefault);
    EXPECT_EQ(out->num_rows(), 0);
    EXPECT_EQ(out->num_columns(), 2);
}

// When no left key is in right, every left row survives.
TEST(AntiJoin, NoMatchPassesEverything) {
    std::vector<gamax::column_ptr> lc;
    lc.push_back(make_int32_col({10, 20, 30}));
    lc.push_back(make_int32_col({1, 2, 3}));
    auto L = mk_table(std::move(lc));

    std::vector<gamax::column_ptr> rc;
    rc.push_back(make_int32_col({1, 2, 3, 4, 5}));  // disjoint from L's keys
    auto R = mk_table(std::move(rc));

    auto out = gamax::join::anti_join(L->view(), R->view(), hash_hint(), cudaStreamDefault);
    EXPECT_EQ(out->num_rows(), 3);
    auto keys = to_host(out->get_column(0).view());
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys, (std::vector<int32_t>{10, 20, 30}));
}

// Duplicate left-side keys: all copies of a matching key drop; all
// copies of a non-matching key pass through unchanged.
TEST(AntiJoin, DuplicateLeftKeysFilterAllCopies) {
    std::vector<gamax::column_ptr> lc;
    lc.push_back(make_int32_col({7, 7, 8, 9, 9, 9, 10}));
    auto L = mk_table(std::move(lc));

    std::vector<gamax::column_ptr> rc;
    rc.push_back(make_int32_col({7, 10}));   // 7 and 10 match; 8, 9 don't
    auto R = mk_table(std::move(rc));

    auto out = gamax::join::anti_join(L->view(), R->view(), hash_hint(), cudaStreamDefault);
    ASSERT_EQ(out->num_rows(), 4);  // one 8, three 9s
    auto keys = to_host(out->get_column(0).view());
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys, (std::vector<int32_t>{8, 9, 9, 9}));
}

// Rejecting right-side duplicates doesn't change the anti-join: only
// membership matters, not multiplicity.
TEST(AntiJoin, RightDuplicatesDoNotAffectResult) {
    std::vector<gamax::column_ptr> lc;
    lc.push_back(make_int32_col({1, 2, 3, 4}));
    auto L = mk_table(std::move(lc));

    std::vector<gamax::column_ptr> rc;
    rc.push_back(make_int32_col({2, 2, 2, 4, 4}));  // 2 appears 3×, 4 appears 2×
    auto R = mk_table(std::move(rc));

    auto out = gamax::join::anti_join(L->view(), R->view(), hash_hint(), cudaStreamDefault);
    auto keys = to_host(out->get_column(0).view());
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys, (std::vector<int32_t>{1, 3}));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
