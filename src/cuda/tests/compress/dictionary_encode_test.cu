#include <iostream>
#include <omp.h>
#include <gtest/gtest.h>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "benchmarks/groupby/groupby_util.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "compress/dictionary_encode.hpp"
#include "types/type_dispatcher.hpp"

#include "data_gen/generator.cuh"

template <typename T, typename EncodedT = T>
void check_correctness(
    T* data, 
    const gamax::size_type n, 
    const gamax::size_type ndv, 
    gamax::column& encoded_col, 
    const gamax::size_type dict_size) 
{
    ASSERT_EQ(n, encoded_col.size());
    ASSERT_EQ(ndv, dict_size);

    auto encoded_data = std::make_unique<EncodedT[]>(n);
    gamax::copy_column_to_host(encoded_col, encoded_data.get());

    std::unordered_map<T, EncodedT> dict;
    for(gamax::size_type i = 0; i < n; i++) {
        if(dict.find(data[i]) == dict.end()) {
            dict[data[i]] = encoded_data[i];
        } else {
            ASSERT_EQ(dict[data[i]], encoded_data[i]);
        }
    }

    std::unordered_map<EncodedT, T> reverse_dict;
    std::vector<bool> seen_dict(dict_size, false); 
    for(gamax::size_type i = 0; i < n; i++) {
        if(!seen_dict[encoded_data[i]]) seen_dict[encoded_data[i]] = true;

        if(reverse_dict.find(encoded_data[i]) == reverse_dict.end()) {
            reverse_dict[encoded_data[i]] = data[i];
        } else {
            ASSERT_EQ(reverse_dict[encoded_data[i]], data[i]);
        }
    }

    ASSERT_TRUE(std::all_of(seen_dict.begin(), seen_dict.end(), [](bool x) { return x; }));
}

template <typename T>
void test_dictionary_encode(const gamax::size_type n, const gamax::size_type ndv) {
    auto data = std::make_unique<T[]>(n);
    for(gamax::size_type i = 0; i < n; i++) {
        data[i] = i % ndv;
    }
    knuth_shuffle(data.get(), n);

    auto col = gamax::make_column_from_host(data.get(), n);
    std::cout << "Data generated\n";

    auto [encoded_col, dict_size] = gamax::compress::dictionary_encode(col->view(), gamax::data_type{gamax::type_to_id<T>()}, ndv, 0.25);

    cudaDeviceSynchronize();

    std::cout << "Dictionary encoded\n";
    check_correctness(data.get(), n, ndv, *encoded_col, dict_size);
}

// Test with different input type T and encoded type EncodedT
template <typename T, typename EncodedT>
void test_dictionary_encode_typed(const gamax::size_type n, const gamax::size_type ndv) {
    auto data = std::make_unique<T[]>(n);
    for(gamax::size_type i = 0; i < n; i++) {
        data[i] = static_cast<T>(i % ndv);
    }
    knuth_shuffle(data.get(), n);

    auto col = gamax::make_column_from_host(data.get(), n);
    std::cout << "Data generated\n";

    auto [encoded_col, dict_size] = gamax::compress::dictionary_encode(
        col->view(), gamax::data_type{gamax::type_to_id<EncodedT>()}, ndv, 0.25);

    cudaDeviceSynchronize();

    std::cout << "Dictionary encoded\n";
    check_correctness<T, EncodedT>(data.get(), n, ndv, *encoded_col, dict_size);
}

TEST(TEST_DICTIONARY_ENCODE, INT32_N4096_NDV1) {
    test_dictionary_encode<int32_t>(4096, 1);
}

TEST(TEST_DICTIONARY_ENCODE, INT32_N4096_NDV16) {
    test_dictionary_encode<int32_t>(4096, 16);
}

TEST(TEST_DICTIONARY_ENCODE, INT32_N4096_NDV4096) {
    test_dictionary_encode<int32_t>(4096, 4096);
}

TEST(TEST_DICTIONARY_ENCODE, INT32_N268435356_NDV4096) {
    test_dictionary_encode<int32_t>((1<<28), 4096);
}

TEST(TEST_DICTIONARY_ENCODE, INT32_N268435356_NDV65536) {
    test_dictionary_encode<int32_t>((1<<28), (1<<16));
}

// Tests with different input and encoded types

// INT64 input with smaller encoded types
TEST(TEST_DICTIONARY_ENCODE, INT64_TO_INT32_N4096_NDV16) {
    test_dictionary_encode_typed<int64_t, int32_t>(4096, 16);
}

TEST(TEST_DICTIONARY_ENCODE, INT64_TO_INT16_N4096_NDV256) {
    test_dictionary_encode_typed<int64_t, int16_t>(4096, 256);
}

TEST(TEST_DICTIONARY_ENCODE, INT64_TO_INT8_N4096_NDV100) {
    test_dictionary_encode_typed<int64_t, int8_t>(4096, 100);
}

// INT32 input with smaller encoded types
TEST(TEST_DICTIONARY_ENCODE, INT32_TO_INT16_N4096_NDV256) {
    test_dictionary_encode_typed<int32_t, int16_t>(4096, 256);
}

TEST(TEST_DICTIONARY_ENCODE, INT32_TO_INT8_N4096_NDV100) {
    test_dictionary_encode_typed<int32_t, int8_t>(4096, 100);
}

// Floating point input with integral encoded types
TEST(TEST_DICTIONARY_ENCODE, FLOAT64_TO_INT32_N4096_NDV16) {
    test_dictionary_encode_typed<double, int32_t>(4096, 16);
}

TEST(TEST_DICTIONARY_ENCODE, FLOAT32_TO_INT16_N4096_NDV256) {
    test_dictionary_encode_typed<float, int16_t>(4096, 256);
}

// Larger scale tests with different types
TEST(TEST_DICTIONARY_ENCODE, INT64_TO_INT32_N1048576_NDV4096) {
    test_dictionary_encode_typed<int64_t, int32_t>(1 << 20, 4096);
}

TEST(TEST_DICTIONARY_ENCODE, INT64_TO_INT16_N1048576_NDV1024) {
    test_dictionary_encode_typed<int64_t, int16_t>(1 << 20, 1024);
}

// ============================================================================
// Order-preserving dictionary encoding tests
// ============================================================================

template <typename T, typename EncodedT = T>
void check_order_correctness(
    T* data, 
    const gamax::size_type n, 
    const gamax::size_type ndv, 
    gamax::column& encoded_col, 
    const gamax::size_type dict_size) 
{
    ASSERT_EQ(n, encoded_col.size());
    ASSERT_EQ(ndv, dict_size);

    auto encoded_data = std::make_unique<EncodedT[]>(n);
    gamax::copy_column_to_host(encoded_col, encoded_data.get());

    // Check basic correctness (same value -> same encoding, different value -> different encoding)
    std::unordered_map<T, EncodedT> dict;
    for(gamax::size_type i = 0; i < n; i++) {
        if(dict.find(data[i]) == dict.end()) {
            dict[data[i]] = encoded_data[i];
        } else {
            ASSERT_EQ(dict[data[i]], encoded_data[i]);
        }
    }

    // Check order preservation: if data[i] > data[j], then encoded[i] > encoded[j]
    // We check this by verifying the mapping is order-preserving
    std::vector<std::pair<T, EncodedT>> value_encoding_pairs;
    for(const auto& [val, enc] : dict) {
        value_encoding_pairs.push_back({val, enc});
    }
    
    // Sort by original value
    std::sort(value_encoding_pairs.begin(), value_encoding_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Check that encoded values are also sorted (0, 1, 2, ...)
    for(gamax::size_type i = 0; i < value_encoding_pairs.size(); i++) {
        ASSERT_EQ(value_encoding_pairs[i].second, static_cast<EncodedT>(i))
            << "Order not preserved: value " << value_encoding_pairs[i].first 
            << " has encoding " << static_cast<int64_t>(value_encoding_pairs[i].second)
            << " but expected " << i;
    }
}

template <typename T>
void test_dictionary_encode_with_order(const gamax::size_type n, const gamax::size_type ndv) {
    auto data = std::make_unique<T[]>(n);
    for(gamax::size_type i = 0; i < n; i++) {
        data[i] = static_cast<T>(i % ndv);
    }
    knuth_shuffle(data.get(), n);

    auto col = gamax::make_column_from_host(data.get(), n);
    std::cout << "Data generated\n";

    auto [encoded_col, dict_size] = gamax::compress::dictionary_encode_with_order(
        col->view(), gamax::data_type{gamax::type_to_id<T>()}, ndv, 0.25);

    cudaDeviceSynchronize();

    std::cout << "Dictionary encoded with order\n";
    check_order_correctness(data.get(), n, ndv, *encoded_col, dict_size);
}

template <typename T, typename EncodedT>
void test_dictionary_encode_with_order_typed(const gamax::size_type n, const gamax::size_type ndv) {
    auto data = std::make_unique<T[]>(n);
    for(gamax::size_type i = 0; i < n; i++) {
        data[i] = static_cast<T>(i % ndv);
    }
    knuth_shuffle(data.get(), n);

    auto col = gamax::make_column_from_host(data.get(), n);
    std::cout << "Data generated\n";

    auto [encoded_col, dict_size] = gamax::compress::dictionary_encode_with_order(
        col->view(), gamax::data_type{gamax::type_to_id<EncodedT>()}, ndv, 0.25);

    cudaDeviceSynchronize();

    std::cout << "Dictionary encoded with order\n";
    check_order_correctness<T, EncodedT>(data.get(), n, ndv, *encoded_col, dict_size);
}

// Basic order-preserving tests
TEST(TEST_DICTIONARY_ENCODE_WITH_ORDER, INT32_N4096_NDV1) {
    test_dictionary_encode_with_order<int32_t>(4096, 1);
}

TEST(TEST_DICTIONARY_ENCODE_WITH_ORDER, INT32_N4096_NDV16) {
    test_dictionary_encode_with_order<int32_t>(4096, 16);
}

TEST(TEST_DICTIONARY_ENCODE_WITH_ORDER, INT32_N4096_NDV4096) {
    test_dictionary_encode_with_order<int32_t>(4096, 4096);
}

TEST(TEST_DICTIONARY_ENCODE_WITH_ORDER, INT32_N1048576_NDV4096) {
    test_dictionary_encode_with_order<int32_t>(1 << 20, 4096);
}

// Order-preserving with different input and encoded types
TEST(TEST_DICTIONARY_ENCODE_WITH_ORDER, INT64_TO_INT32_N4096_NDV16) {
    test_dictionary_encode_with_order_typed<int64_t, int32_t>(4096, 16);
}

TEST(TEST_DICTIONARY_ENCODE_WITH_ORDER, INT32_TO_INT16_N4096_NDV256) {
    test_dictionary_encode_with_order_typed<int32_t, int16_t>(4096, 256);
}

TEST(TEST_DICTIONARY_ENCODE_WITH_ORDER, INT64_TO_INT16_N1048576_NDV1024) {
    test_dictionary_encode_with_order_typed<int64_t, int16_t>(1 << 20, 1024);
}

// Floating point with order preservation
TEST(TEST_DICTIONARY_ENCODE_WITH_ORDER, FLOAT64_TO_INT32_N4096_NDV16) {
    test_dictionary_encode_with_order_typed<double, int32_t>(4096, 16);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}