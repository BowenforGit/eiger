// Unit tests for phys::distinct — hash-based dedup over input key
// columns. Covers empty input, all-unique, all-duplicate, mixed
// numeric, multi-key numeric, STRING, and multi-key STRING+numeric
// shapes. Row count is always checked exactly; row contents are
// checked as a *set* because insertion order (and therefore row
// order in the output) is implementation-defined — callers that
// need a specific order sort downstream.

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/groupby.hpp"
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
    return gamax::make_column_from_host(
        const_cast<T*>(v.data()),
        static_cast<gamax::size_type>(v.size()));
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

// Dump an Arrow STRING column back to host as a std::vector<std::string>.
// Offsets are INT64 in this codebase (see get_arrow_string_offsets).
std::vector<std::string> dump_string_column(
    gamax::column_view col,
    cudaStream_t stream = cudaStreamDefault)
{
    gamax::string_column_view sv(col, gamax::ARROW_STRING, stream);
    const auto n = col.size();
    std::vector<gamax::size_type> offsets(n + 1);
    cudaMemcpyAsync(offsets.data(), sv.meta(),
                    (n + 1) * sizeof(gamax::size_type),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const gamax::size_type total = offsets.back();
    std::string flat(total, '\0');
    if (total > 0) {
        cudaMemcpyAsync(flat.data(),
                        const_cast<void*>(sv.data<void>()),
                        total, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    std::vector<std::string> out;
    out.reserve(n);
    for (gamax::size_type i = 0; i < n; ++i) {
        out.emplace_back(flat.data() + offsets[i],
                         offsets[i + 1] - offsets[i]);
    }
    return out;
}

} // namespace

TEST(PhysDistinct, EmptyInputYieldsEmptyOutput) {
    std::vector<int32_t> v;
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(v));
    auto tbl = gamax::make_table(std::move(cols));
    auto out = phys::distinct(tbl->view(), default_ctx());
    EXPECT_EQ(out->num_rows(), 0);
    ASSERT_EQ(out->num_columns(), 1);
    EXPECT_EQ(out->view()[0].type().id(), type_id::INT32);
}

TEST(PhysDistinct, AllUniqueIsIdentityInShape) {
    std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));
    auto out = phys::distinct(tbl->view(), default_ctx());
    ASSERT_EQ(out->num_rows(), 10);
    auto host = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    std::set<int32_t> as_set(host.get(), host.get() + 10);
    std::set<int32_t> expected(a.begin(), a.end());
    EXPECT_EQ(as_set, expected);
}

TEST(PhysDistinct, AllDuplicatesCollapseToOneRow) {
    std::vector<int32_t> a = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));
    auto out = phys::distinct(tbl->view(), default_ctx());
    ASSERT_EQ(out->num_rows(), 1);
    auto host = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    EXPECT_EQ(host[0], 7);
}

TEST(PhysDistinct, Int32ThreeOfTenUnique) {
    // 10 rows, 3 unique values.
    std::vector<int32_t> a = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));
    auto out = phys::distinct(tbl->view(), default_ctx());
    ASSERT_EQ(out->num_rows(), 3);
    auto host = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    std::set<int32_t> as_set(host.get(), host.get() + 3);
    std::set<int32_t> expected = {1, 2, 3};
    EXPECT_EQ(as_set, expected);
}

TEST(PhysDistinct, MultiKeyInt32AndInt64FourOfEight) {
    // 8 rows: pairs (k1, k2) produce 3 unique combinations.
    std::vector<int32_t> k1 = {1, 2, 1, 2, 1, 2, 3, 3};
    std::vector<int64_t> k2 = {10, 20, 10, 20, 10, 20, 30, 30};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(k1));
    cols.push_back(mk_col(k2));
    auto tbl = gamax::make_table(std::move(cols));
    auto out = phys::distinct(tbl->view(), default_ctx());
    ASSERT_EQ(out->num_rows(), 3);  // (1,10), (2,20), (3,30)
    ASSERT_EQ(out->num_columns(), 2);
    auto h1 = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    auto h2 = gamax::copy_column_to_host<int64_t>(out->view()[1]);
    std::set<std::tuple<int32_t, int64_t>> as_set;
    for (int i = 0; i < 3; ++i) as_set.emplace(h1[i], h2[i]);
    std::set<std::tuple<int32_t, int64_t>> expected = {
        {1, 10}, {2, 20}, {3, 30}};
    EXPECT_EQ(as_set, expected);
}

TEST(PhysDistinct, StringKeysTwoOfFive) {
    // 5 rows, 2 unique STRING values.
    std::vector<std::string> brands = {
        "Brand#11", "Brand#22", "Brand#11", "Brand#22", "Brand#11",
    };
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_string_col(brands));
    auto tbl = gamax::make_table(std::move(cols));
    auto out = phys::distinct(tbl->view(), default_ctx());
    EXPECT_EQ(out->num_rows(), 2);
    auto host = dump_string_column(out->view()[0]);
    std::set<std::string> as_set(host.begin(), host.end());
    std::set<std::string> expected = {"Brand#11", "Brand#22"};
    EXPECT_EQ(as_set, expected);
}

TEST(PhysDistinct, MultiKeyStringAndInt32Q16Shape) {
    // Q16-ish: (p_brand, p_type, p_size) — simplified to 2 STRING
    // cols + 1 INT. 20 rows, 4 unique (brand, type, size) triples.
    std::vector<std::string> brand;
    std::vector<std::string> type_;
    std::vector<int32_t> size;
    for (int i = 0; i < 20; ++i) {
        brand.push_back((i % 4 < 2) ? "Brand#11" : "Brand#22");
        type_.push_back((i % 4 % 2 == 0) ? "SMALL POLISHED COPPER"
                                          : "LARGE ANODIZED STEEL");
        size.push_back(i % 4);
    }
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_string_col(brand));
    cols.push_back(mk_string_col(type_));
    cols.push_back(mk_col(size));
    auto tbl = gamax::make_table(std::move(cols));
    auto out = phys::distinct(tbl->view(), default_ctx());
    ASSERT_EQ(out->num_rows(), 4);
    ASSERT_EQ(out->num_columns(), 3);
    auto b = dump_string_column(out->view()[0]);
    auto ty = dump_string_column(out->view()[1]);
    auto s = gamax::copy_column_to_host<int32_t>(out->view()[2]);
    std::set<std::tuple<std::string, std::string, int32_t>> as_set;
    for (int i = 0; i < 4; ++i) {
        as_set.emplace(b[i], ty[i], s[i]);
    }
    // The 4 unique triples produced by the constructor above:
    //   (Brand#11, SMALL...COPPER, 0)
    //   (Brand#11, LARGE...STEEL,  1)
    //   (Brand#22, SMALL...COPPER, 2)
    //   (Brand#22, LARGE...STEEL,  3)
    std::set<std::tuple<std::string, std::string, int32_t>> expected = {
        {"Brand#11", "SMALL POLISHED COPPER", 0},
        {"Brand#11", "LARGE ANODIZED STEEL", 1},
        {"Brand#22", "SMALL POLISHED COPPER", 2},
        {"Brand#22", "LARGE ANODIZED STEEL", 3},
    };
    EXPECT_EQ(as_set, expected);
}

// ── Sort-based distinct ───────────────────────────────────────────
//
// `phys::distinct` dispatches on `hint.groupby_algo`. Cover the
// single-col path (Case 1) and the multi-col + SKF path (Case 2),
// the all-unique fast path, and the two throw branches (multi-col
// without SKF, STRING).

namespace {
gamax::groupby::groupby_hint sort_hint(bool enable_skf) {
    gamax::groupby::groupby_hint h;
    h.groupby_algo = gamax::groupby::SORT;
    h.enable_skf = enable_skf;
    return h;
}
} // namespace

TEST(PhysDistinctSort, SingleColInt32ThreeOfTen) {
    std::vector<int32_t> a = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));
    auto hint = sort_hint(/*enable_skf=*/false);
    auto out = phys::distinct(tbl->view(), default_ctx(), hint);
    ASSERT_EQ(out->num_rows(), 3);
    auto host = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    std::set<int32_t> as_set(host.get(), host.get() + 3);
    EXPECT_EQ(as_set, (std::set<int32_t>{1, 2, 3}));
}

TEST(PhysDistinctSort, SingleColInt64AllUnique) {
    // No duplicates: RLE produces num_runs == n. (Case 1 has no
    // explicit fast path — the RLE just emits each value once.)
    std::vector<int64_t> a = {7, 13, 42, 99, 100, 1000};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(a));
    auto tbl = gamax::make_table(std::move(cols));
    auto hint = sort_hint(/*enable_skf=*/false);
    auto out = phys::distinct(tbl->view(), default_ctx(), hint);
    ASSERT_EQ(out->num_rows(), 6);
    auto host = gamax::copy_column_to_host<int64_t>(out->view()[0]);
    std::set<int64_t> as_set(host.get(), host.get() + 6);
    EXPECT_EQ(as_set, (std::set<int64_t>{7, 13, 42, 99, 100, 1000}));
}

TEST(PhysDistinctSort, MultiKeyInt32Int64WithSKF) {
    // Same shape as the hash-side MultiKeyInt32AndInt64FourOfEight
    // test (3 unique (k1, k2) combos in 8 rows), but routed through
    // the SORT + SKF path. multikey_compress fuses two narrow ints
    // into a single uint64 key; sort distinct then RLEs the fused
    // key and materialises original-key rows via leader_tid.
    std::vector<int32_t> k1 = {1, 2, 1, 2, 1, 2, 3, 3};
    std::vector<int64_t> k2 = {10, 20, 10, 20, 10, 20, 30, 30};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(k1));
    cols.push_back(mk_col(k2));
    auto tbl = gamax::make_table(std::move(cols));
    auto hint = sort_hint(/*enable_skf=*/true);
    auto out = phys::distinct(tbl->view(), default_ctx(), hint);
    ASSERT_EQ(out->num_rows(), 3);
    ASSERT_EQ(out->num_columns(), 2);
    auto h1 = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    auto h2 = gamax::copy_column_to_host<int64_t>(out->view()[1]);
    std::set<std::tuple<int32_t, int64_t>> as_set;
    for (int i = 0; i < 3; ++i) as_set.emplace(h1[i], h2[i]);
    EXPECT_EQ(as_set,
              (std::set<std::tuple<int32_t, int64_t>>{
                  {1, 10}, {2, 20}, {3, 30}}));
}

TEST(PhysDistinctSort, MultiKeyAllUniqueFastPath) {
    // Exercise Case 2's fast path: when every (k1, k2) is distinct,
    // num_runs == n and the implementation returns a column-wise copy
    // of the original input.
    std::vector<int32_t> k1 = {1, 2, 3, 4};
    std::vector<int64_t> k2 = {10, 20, 30, 40};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(k1));
    cols.push_back(mk_col(k2));
    auto tbl = gamax::make_table(std::move(cols));
    auto hint = sort_hint(/*enable_skf=*/true);
    auto out = phys::distinct(tbl->view(), default_ctx(), hint);
    ASSERT_EQ(out->num_rows(), 4);
    ASSERT_EQ(out->num_columns(), 2);
    auto h1 = gamax::copy_column_to_host<int32_t>(out->view()[0]);
    auto h2 = gamax::copy_column_to_host<int64_t>(out->view()[1]);
    std::set<std::tuple<int32_t, int64_t>> as_set;
    for (int i = 0; i < 4; ++i) as_set.emplace(h1[i], h2[i]);
    EXPECT_EQ(as_set,
              (std::set<std::tuple<int32_t, int64_t>>{
                  {1, 10}, {2, 20}, {3, 30}, {4, 40}}));
}

TEST(PhysDistinctSort, MultiKeyWithoutSKFThrows) {
    // Multi-col input without SKF — sort distinct is single-key-only,
    // so this must throw with a message that points the caller at the
    // remedy.
    std::vector<int32_t> k1 = {1, 2, 1, 2};
    std::vector<int64_t> k2 = {10, 20, 10, 20};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_col(k1));
    cols.push_back(mk_col(k2));
    auto tbl = gamax::make_table(std::move(cols));
    auto hint = sort_hint(/*enable_skf=*/false);
    EXPECT_THROW(
        phys::distinct(tbl->view(), default_ctx(), hint),
        std::runtime_error);
}

TEST(PhysDistinctSort, StringKeyThrows) {
    // STRING keys are explicitly rejected by sort::distinct's entry
    // guard — radix sort doesn't accept them.
    std::vector<std::string> brands = {"Brand#11", "Brand#22", "Brand#11"};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(mk_string_col(brands));
    auto tbl = gamax::make_table(std::move(cols));
    auto hint = sort_hint(/*enable_skf=*/false);
    EXPECT_THROW(
        phys::distinct(tbl->view(), default_ctx(), hint),
        std::runtime_error);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
