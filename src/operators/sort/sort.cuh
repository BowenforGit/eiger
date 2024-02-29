#pragma once

#define CUB_STDERR

#include <iostream>
#include <tuple>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 

#include "types/tuple.cuh"
#include "utils/utils.cuh"

namespace gamax {
    namespace sort {
        template <typename T, typename NumItemsT>
        void radix_sort_k(T* d_in, 
                          T* d_out, 
                          NumItemsT num_items, 
                          int begin_bit = 0, 
                          int end_bit = sizeof(T) * 8, 
                          void *d_temp_storage = nullptr,
                          size_t temp_storage_bytes = 0, 
                          cudaStream_t stream = 0) {
            bool alloc_temp_storage = (d_temp_storage == nullptr);
            if (alloc_temp_storage) {
                cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_in, d_out, num_items, begin_bit, end_bit, stream);
                allocate_mem(&d_temp_storage, false, temp_storage_bytes);
            }
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, begin_bit, end_bit, stream);
            if (alloc_temp_storage) {
                release_mem(d_temp_storage);
            }
        }

        template <typename KeyT, typename ValueT, typename NumItemsT>
        void radix_sort_kv(KeyT* d_kin, 
                          KeyT* d_kout, 
                          ValueT* d_vin,
                          ValueT* d_vout,
                          NumItemsT num_items, 
                          int begin_bit = 0, 
                          int end_bit = sizeof(KeyT) * 8, 
                          void *d_temp_storage = nullptr,
                          size_t temp_storage_bytes = 0, 
                          cudaStream_t stream = 0) {
            bool alloc_temp_storage = (d_temp_storage == nullptr);
            if (alloc_temp_storage) {
                cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_kin, d_kout, d_vin, d_vout, num_items, begin_bit, end_bit, stream);
                allocate_mem(&d_temp_storage, false, temp_storage_bytes);
            }
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_kin, d_kout, d_vin, d_vout, num_items, begin_bit, end_bit, stream);
            if (alloc_temp_storage) {
                release_mem(d_temp_storage);
            }
        }
    } // namespace sort
} // namespace gamax