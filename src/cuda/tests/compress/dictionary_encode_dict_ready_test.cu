// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <iostream>
#include <omp.h>
#include <gtest/gtest.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "compress/dictionary_encode.hpp"
#include "operators/sort/sort.hpp"

template <typename Key, typename Encode>
void check_correctness(
    gamax::column& key,
    gamax::column& res,
    const gamax::size_type dict_size)
{
    auto n = key.size();

    ASSERT_EQ(n, res.size());

    auto data = std::make_unique<Key[]>(n);
    auto encoded_data = std::make_unique<Encode[]>(n);
    gamax::copy_column_to_host(key, data.get(), cudaStreamDefault);
    gamax::copy_column_to_host(res, encoded_data.get(), cudaStreamDefault);

    std::unordered_map<Key, Encode> dict;
    for(gamax::size_type i = 0; i < n; i++) {
        ASSERT_TRUE(data[i] < dict_size);
        if(dict.find(data[i]) == dict.end()) {
            dict[data[i]] = encoded_data[i];
        } else {
            ASSERT_EQ(dict[data[i]], encoded_data[i]);
        }
    }

    std::unordered_map<Encode, Key> reverse_dict;
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

template <typename Key, typename Encode>
void test_dictionary_encode(const gamax::size_type num_items, const gamax::size_type dict_size) {
    constexpr cudaStream_t stream = cudaStreamDefault;
    auto dict = gamax::make_sequence<Key>(dict_size, stream, 0, 1);
    auto keys = gamax::make_sequence<Key>(num_items, stream, 0, 1);

    auto raw_key_ptr = keys->template data<Key>();
    
    auto policy = gamax::rmm_exec_policy_handle{stream};
    thrust::for_each_n(policy.get(), raw_key_ptr, num_items, [=] __device__ (Key& key) { key = key % dict_size; });
    thrust::default_random_engine g;
    thrust::shuffle(policy.get(), raw_key_ptr, raw_key_ptr + num_items, g);
    cudaDeviceSynchronize();

    auto res = 
        gamax::compress::dictionary_encode(
            keys->view(), 
            dict->view(), 
            stream,
            gamax::data_type{gamax::type_to_id<Encode>()});
    cudaDeviceSynchronize();

    // auto values = gamax::make_sequence<int32_t>(num_items, 0, 1);
    // gamax::sort::sort_column_pair(*res, *values, 0, 12);
    // gamax::sort::sort_column_pair(*keys, *values);

    std::cout << "Dictionary encoded\n";
    check_correctness<Key, Encode>(*keys, *res, dict_size);
}

#define TEST_DICTIONARY_ENCODE(KEY_TYPE, ENCODE_TYPE, NUM_ITEMS, DICT_SIZE) \
    TEST(TEST_DICTIONARY_ENCODE, KEY_TYPE ## _N ## NUM_ITEMS ## _NDV ## DICT_SIZE) { \
        test_dictionary_encode<KEY_TYPE, ENCODE_TYPE>(NUM_ITEMS, DICT_SIZE); \
    }

TEST_DICTIONARY_ENCODE(int32_t, int32_t, 4096, 1)
TEST_DICTIONARY_ENCODE(int32_t, int32_t, 4096, 16)
TEST_DICTIONARY_ENCODE(int32_t, int32_t, 4096, 4096)
TEST_DICTIONARY_ENCODE(int32_t, int32_t, 268435356, 1024) // 2^28
TEST_DICTIONARY_ENCODE(int32_t, int32_t, 268435356, 2048) // 2^28
TEST_DICTIONARY_ENCODE(int32_t, int32_t, 268435356, 4096) // 2^28
TEST_DICTIONARY_ENCODE(int32_t, int32_t, 536870712, 1024) // 2^29

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
