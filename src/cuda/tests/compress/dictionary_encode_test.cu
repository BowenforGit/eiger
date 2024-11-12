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

#include "data_gen/generator.cuh"

template <typename T>
void check_correctness(
    T* data, 
    const gamax::size_type n, 
    const gamax::size_type ndv, 
    gamax::column& encoded_col, 
    const gamax::size_type dict_size) 
{
    ASSERT_EQ(n, encoded_col.size());
    ASSERT_EQ(ndv, dict_size);

    auto encoded_data = std::make_unique<gamax::size_type[]>(n);
    gamax::copy_column_to_host(encoded_col, encoded_data.get());

    std::unordered_map<T, gamax::size_type> dict;
    for(gamax::size_type i = 0; i < n; i++) {
        if(dict.find(data[i]) == dict.end()) {
            dict[data[i]] = encoded_data[i];
        } else {
            ASSERT_EQ(dict[data[i]], encoded_data[i]);
        }
    }

    std::unordered_map<gamax::size_type, T> reverse_dict;
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

    auto [encoded_col, dict_size] = gamax::compress::dictionary_encode(*col, ndv, 0.5);

    cudaDeviceSynchronize();

    std::cout << "Dictionary encoded\n";
    check_correctness(data.get(), n, ndv, *encoded_col, dict_size);
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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}