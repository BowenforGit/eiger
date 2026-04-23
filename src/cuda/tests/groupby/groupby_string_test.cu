// Inline sanity tests for STRING keys in hash-based groupby.
// Ground truth is small enough to verify by hand; see comments below.
//
// What this file exercises that the existing tests don't:
//   - STRING keys (single-column, multi-column, mixed with numeric)
//   - Strings that share prefixes (regression guard for compare-as-
//     startsWith bugs)
//   - Long strings that exceed any inlined-header prefix
//   - Empty strings
//   - Per-group value verification (not just a sum of all group sums)
//   - MIN / MAX aggregation on numeric values alongside STRING keys
//   - Larger group cardinality to stress the hash set

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"

namespace {

// Build a STRING column from a vector of host strings using the
// flat-data + lengths constructor of `make_string_column`.
gamax::column_ptr make_string_col(const std::vector<std::string>& strs,
                                  cudaStream_t stream = cudaStreamDefault)
{
    std::string flat;
    std::vector<int32_t> lengths;
    lengths.reserve(strs.size());
    for (const auto& s : strs) {
        flat.append(s);
        lengths.push_back(static_cast<int32_t>(s.size()));
    }
    auto sc = gamax::make_string_column(
        flat.data(),
        lengths.data(),
        static_cast<gamax::size_type>(strs.size()),
        stream);
    return gamax::column_ptr{std::move(sc)};
}

gamax::column_ptr make_int64_col(const std::vector<int64_t>& vs)
{
    return gamax::make_column_from_host(
        const_cast<int64_t*>(vs.data()),
        static_cast<gamax::size_type>(vs.size()));
}

// Pull a STRING column's rows back to host as std::strings so tests can
// match expected per-group values independent of the group ordering.
std::vector<std::string> read_strings_to_host(gamax::column_view col,
                                              cudaStream_t stream = cudaStreamDefault)
{
    gamax::string_column_view sv(col, gamax::ARROW_STRING, stream);
    const auto n = sv.size();
    std::vector<gamax::size_type> offsets(n + 1);
    cudaMemcpyAsync(offsets.data(), sv.meta(),
                    (n + 1) * sizeof(gamax::size_type),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const auto total_bytes = offsets.back();
    std::vector<char> bytes(total_bytes);
    if (total_bytes > 0) {
        cudaMemcpyAsync(bytes.data(), sv.template data<char>(),
                        total_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    std::vector<std::string> out;
    out.reserve(n);
    for (gamax::size_type i = 0; i < n; ++i) {
        out.emplace_back(bytes.data() + offsets[i],
                         bytes.data() + offsets[i + 1]);
    }
    return out;
}

std::vector<int64_t> read_int64_to_host(gamax::column_view col)
{
    auto h = gamax::copy_column_to_host<int64_t>(col);
    std::vector<int64_t> out(col.size());
    for (gamax::size_type i = 0; i < col.size(); ++i) out[i] = h[i];
    return out;
}

} // namespace

// ── Existing coverage ─────────────────────────────────────────────

// Single STRING key, SUM of INT64 values.
//
//   keys: A, A, B, A, B
//   vals: 1, 2, 3, 4, 5
//   expected: A→7, B→8
TEST(GROUPBY_STRING, SingleKeySumInt64) {
    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col({"A", "A", "B", "A", "B"}));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vals = std::vector<gamax::column_ptr>{};
    vals.push_back(make_int64_col({1, 2, 3, 4, 5}));
    auto val_tbl = gamax::make_table(std::move(vals));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 2);
    ASSERT_EQ(out->num_columns(), 2);

    auto out_keys = read_strings_to_host(out->get_column(0).view());
    auto out_sums = read_int64_to_host(out->get_column(1).view());
    std::unordered_map<std::string, int64_t> got;
    for (size_t i = 0; i < out_keys.size(); ++i) got[out_keys[i]] = out_sums[i];
    EXPECT_EQ(got["A"], 7);
    EXPECT_EQ(got["B"], 8);
}

// Mixed {STRING, INT64} keys, SUM of INT64 values.
//
//   keys: ("A", 1), ("A", 1), ("B", 1), ("A", 2), ("B", 1)
//   vals:    10,      20,       30,       40,       50
//   expected: (A,1)→30, (A,2)→40, (B,1)→80
TEST(GROUPBY_STRING, MixedStringAndIntKeySumInt64) {
    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col({"A", "A", "B", "A", "B"}));
    keys.push_back(make_int64_col({1, 1, 1, 2, 1}));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vals = std::vector<gamax::column_ptr>{};
    vals.push_back(make_int64_col({10, 20, 30, 40, 50}));
    auto val_tbl = gamax::make_table(std::move(vals));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 3);
    ASSERT_EQ(out->num_columns(), 3);

    auto k0 = read_strings_to_host(out->get_column(0).view());
    auto k1 = read_int64_to_host(out->get_column(1).view());
    auto s  = read_int64_to_host(out->get_column(2).view());
    std::unordered_map<std::string, int64_t> got;
    for (size_t i = 0; i < k0.size(); ++i) {
        got[k0[i] + "|" + std::to_string(k1[i])] = s[i];
    }
    EXPECT_EQ(got["A|1"], 30);
    EXPECT_EQ(got["A|2"], 40);
    EXPECT_EQ(got["B|1"], 80);
}

// ── New coverage ──────────────────────────────────────────────────

// Per-group value verification. Catches bugs where total SUM agrees but
// the key/value correspondence is broken (e.g. keys gathered from a
// wrong tid map).
//
//   keys: 3-MEDIUM, 1-URGENT, 3-MEDIUM, 2-HIGH, 1-URGENT, 3-MEDIUM
//   vals:     100,       10,      200,     30,       20,      300
//   expected: 1-URGENT→30, 2-HIGH→30, 3-MEDIUM→600
TEST(GROUPBY_STRING, PerGroupValueIdentity) {
    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col({
        "3-MEDIUM", "1-URGENT", "3-MEDIUM",
        "2-HIGH",   "1-URGENT", "3-MEDIUM",
    }));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vals = std::vector<gamax::column_ptr>{};
    vals.push_back(make_int64_col({100, 10, 200, 30, 20, 300}));
    auto val_tbl = gamax::make_table(std::move(vals));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 3);
    auto out_keys = read_strings_to_host(out->get_column(0).view());
    auto out_sums = read_int64_to_host(out->get_column(1).view());
    std::unordered_map<std::string, int64_t> got;
    for (size_t i = 0; i < out_keys.size(); ++i) got[out_keys[i]] = out_sums[i];
    EXPECT_EQ(got["1-URGENT"], 30);
    EXPECT_EQ(got["2-HIGH"],   30);
    EXPECT_EQ(got["3-MEDIUM"], 600);
}

// Prefix-overlap regression. If STRING equality falls through to a
// "starts-with" check, rows with key "abc" would be treated as equal to
// rows with key "abcd", collapsing three distinct keys into one.
TEST(GROUPBY_STRING, PrefixOverlapKeysAreDistinct) {
    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col({
        "abc", "abcd", "abcde", "abc", "abcd", "abcde",
        "abc", "abcd", "abcde",
    }));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vals = std::vector<gamax::column_ptr>{};
    vals.push_back(make_int64_col({1, 2, 3, 10, 20, 30, 100, 200, 300}));
    auto val_tbl = gamax::make_table(std::move(vals));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 3);
    auto k = read_strings_to_host(out->get_column(0).view());
    auto s = read_int64_to_host(out->get_column(1).view());
    std::unordered_map<std::string, int64_t> got;
    for (size_t i = 0; i < k.size(); ++i) got[k[i]] = s[i];
    EXPECT_EQ(got["abc"],   111);
    EXPECT_EQ(got["abcd"],  222);
    EXPECT_EQ(got["abcde"], 333);
}

// Long keys: 64+ characters. Makes sure the comparator and hasher
// traverse the full string, not just an inlined prefix.
TEST(GROUPBY_STRING, LongKeysBeyondPrefix) {
    // Two strings that share the first 60 chars, diverge at char 61.
    const std::string prefix(60, 'x');
    const std::string ka = prefix + "ALPHA_TAIL_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    const std::string kb = prefix + "BRAVO_TAIL_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    // And a completely different long string.
    const std::string kc(96, 'q');

    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col({ka, kb, ka, kc, kb, kc, ka}));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vals = std::vector<gamax::column_ptr>{};
    vals.push_back(make_int64_col({1, 2, 3, 4, 5, 6, 7}));
    auto val_tbl = gamax::make_table(std::move(vals));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 3);
    auto k = read_strings_to_host(out->get_column(0).view());
    auto s = read_int64_to_host(out->get_column(1).view());
    std::unordered_map<std::string, int64_t> got;
    for (size_t i = 0; i < k.size(); ++i) got[k[i]] = s[i];
    EXPECT_EQ(got[ka], 1 + 3 + 7);
    EXPECT_EQ(got[kb], 2 + 5);
    EXPECT_EQ(got[kc], 4 + 6);
}

// Two STRING keys together (no numeric). Exercises row_hasher's
// multi-string hash-combining path.
//
//   keys: (red,  cat), (red, dog), (blue, cat),
//         (red,  cat), (blue,dog), (blue, cat)
//   vals:     1,           2,          4,
//             8,           16,         32
//   expected: (red,cat)→9, (red,dog)→2, (blue,cat)→36, (blue,dog)→16
TEST(GROUPBY_STRING, TwoStringKeys) {
    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col({"red", "red", "blue", "red", "blue", "blue"}));
    keys.push_back(make_string_col({"cat", "dog", "cat",  "cat", "dog",  "cat"}));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vals = std::vector<gamax::column_ptr>{};
    vals.push_back(make_int64_col({1, 2, 4, 8, 16, 32}));
    auto val_tbl = gamax::make_table(std::move(vals));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 4);
    auto c0 = read_strings_to_host(out->get_column(0).view());
    auto c1 = read_strings_to_host(out->get_column(1).view());
    auto s  = read_int64_to_host(out->get_column(2).view());
    std::unordered_map<std::string, int64_t> got;
    for (size_t i = 0; i < c0.size(); ++i) {
        got[c0[i] + "|" + c1[i]] = s[i];
    }
    EXPECT_EQ(got["red|cat"],  9);
    EXPECT_EQ(got["red|dog"],  2);
    EXPECT_EQ(got["blue|cat"], 36);
    EXPECT_EQ(got["blue|dog"], 16);
}

// Empty strings alongside non-empty keys. Empty-string rows should form
// their own group (not collapse with any other).
TEST(GROUPBY_STRING, EmptyStringKeys) {
    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col({"", "x", "", "x", "y", "", "y"}));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vals = std::vector<gamax::column_ptr>{};
    vals.push_back(make_int64_col({1, 2, 3, 4, 5, 6, 7}));
    auto val_tbl = gamax::make_table(std::move(vals));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 3);
    auto k = read_strings_to_host(out->get_column(0).view());
    auto s = read_int64_to_host(out->get_column(1).view());
    std::unordered_map<std::string, int64_t> got;
    for (size_t i = 0; i < k.size(); ++i) got[k[i]] = s[i];
    EXPECT_EQ(got[""],  1 + 3 + 6);
    EXPECT_EQ(got["x"], 2 + 4);
    EXPECT_EQ(got["y"], 5 + 7);
}

// SUM, MIN, and MAX aggregations over the same numeric column keyed by
// a STRING. Catches bugs where the STRING key-gather disagrees with the
// aggregation kernel's per-group mapping.
TEST(GROUPBY_STRING, MultipleAggregationsMinMaxSum) {
    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col({
        "FOO", "BAR", "FOO", "BAR", "FOO", "BAZ", "BAR", "BAZ",
    }));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vals = std::vector<gamax::column_ptr>{};
    vals.push_back(make_int64_col({5, 3, 8, 7, 2, 100, 1, 50}));
    auto val_tbl = gamax::make_table(std::move(vals));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);
    requests.emplace_back(0, gamax::groupby::aggregation_id::MIN);
    requests.emplace_back(0, gamax::groupby::aggregation_id::MAX);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 3);
    ASSERT_EQ(out->num_columns(), 4);

    auto k   = read_strings_to_host(out->get_column(0).view());
    auto sum = read_int64_to_host(out->get_column(1).view());
    auto mn  = read_int64_to_host(out->get_column(2).view());
    auto mx  = read_int64_to_host(out->get_column(3).view());

    struct Triple { int64_t sum; int64_t mn; int64_t mx; };
    std::unordered_map<std::string, Triple> got;
    for (size_t i = 0; i < k.size(); ++i) got[k[i]] = {sum[i], mn[i], mx[i]};

    EXPECT_EQ(got["FOO"].sum, 15); EXPECT_EQ(got["FOO"].mn, 2); EXPECT_EQ(got["FOO"].mx, 8);
    EXPECT_EQ(got["BAR"].sum, 11); EXPECT_EQ(got["BAR"].mn, 1); EXPECT_EQ(got["BAR"].mx, 7);
    EXPECT_EQ(got["BAZ"].sum, 150); EXPECT_EQ(got["BAZ"].mn, 50); EXPECT_EQ(got["BAZ"].mx, 100);
}

// Larger-cardinality STRING groupby. 1000 rows, ~100 distinct keys,
// each key appears ~10× on average. Verifies hash-set growth, gather
// correctness under many groups, and that the per-group SUM matches
// CPU-computed ground truth for every group.
TEST(GROUPBY_STRING, ManyGroupsLargerInput) {
    constexpr int NUM_ROWS = 1000;
    constexpr int NUM_DISTINCT = 100;
    std::vector<std::string> key_strs;
    std::vector<int64_t>     vals;
    key_strs.reserve(NUM_ROWS);
    vals.reserve(NUM_ROWS);
    std::unordered_map<std::string, int64_t> cpu_sums;
    // Deterministic pseudo-random mixing: row i maps to key i*37 mod NUM_DISTINCT.
    for (int i = 0; i < NUM_ROWS; ++i) {
        int key_id = (i * 37) % NUM_DISTINCT;
        std::string k = "key_" + std::string(key_id % 7 + 1, 'A' + (key_id % 26))
                      + "_" + std::to_string(key_id);
        key_strs.push_back(k);
        int64_t v = static_cast<int64_t>(i);
        vals.push_back(v);
        cpu_sums[k] += v;
    }

    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col(key_strs));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vs = std::vector<gamax::column_ptr>{};
    vs.push_back(make_int64_col(vals));
    auto val_tbl = gamax::make_table(std::move(vs));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(static_cast<size_t>(out->num_rows()), cpu_sums.size());

    auto out_keys = read_strings_to_host(out->get_column(0).view());
    auto out_sums = read_int64_to_host(out->get_column(1).view());

    ASSERT_EQ(out_keys.size(), cpu_sums.size());
    // Every output key is expected and its sum matches the CPU total.
    for (size_t i = 0; i < out_keys.size(); ++i) {
        auto it = cpu_sums.find(out_keys[i]);
        ASSERT_NE(it, cpu_sums.end()) << "unexpected key: " << out_keys[i];
        EXPECT_EQ(out_sums[i], it->second) << "sum mismatch for key: " << out_keys[i];
    }
}

// Extreme skew: one key dominates the input. Verifies the hash set
// doesn't mis-identify the dominant key as new repeatedly (which would
// blow up group count), and that the per-group SUM is correct under
// many collisions on a single slot.
TEST(GROUPBY_STRING, SkewedSingleDominantKey) {
    constexpr int NUM_ROWS = 1000;
    std::vector<std::string> key_strs;
    std::vector<int64_t>     vals;
    key_strs.reserve(NUM_ROWS);
    vals.reserve(NUM_ROWS);
    int64_t expected_dom = 0;
    int64_t expected_other = 0;
    for (int i = 0; i < NUM_ROWS; ++i) {
        if (i % 50 == 0) {
            key_strs.push_back("rare");
            expected_other += i;
        } else {
            key_strs.push_back("dominant");
            expected_dom += i;
        }
        vals.push_back(static_cast<int64_t>(i));
    }

    auto keys = std::vector<gamax::column_ptr>{};
    keys.push_back(make_string_col(key_strs));
    auto key_tbl = gamax::make_table(std::move(keys));

    auto vs = std::vector<gamax::column_ptr>{};
    vs.push_back(make_int64_col(vals));
    auto val_tbl = gamax::make_table(std::move(vs));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.emplace_back(0, gamax::groupby::aggregation_id::SUM);

    gamax::groupby::groupby_hint hint;
    auto out = gamax::groupby::groupby(
        key_tbl->view(), val_tbl->view(), requests, hint);

    ASSERT_EQ(out->num_rows(), 2);
    auto k = read_strings_to_host(out->get_column(0).view());
    auto s = read_int64_to_host(out->get_column(1).view());
    std::unordered_map<std::string, int64_t> got;
    for (size_t i = 0; i < k.size(); ++i) got[k[i]] = s[i];
    EXPECT_EQ(got["dominant"], expected_dom);
    EXPECT_EQ(got["rare"],     expected_other);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
