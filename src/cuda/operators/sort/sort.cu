// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <memory>
#include <optional>
#include <utility> // std::pair

#include <cuda.h>
#include <cub/cub.cuh> 

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "operators/sort/sort.hpp"
#include "memory/column.hpp"
#include "memory/device_buffer.hpp"

namespace gamax::sort {
namespace {
    template <typename T, typename NumItemsT = gamax::size_type>
    void radix_sort_k(const T* d_in, 
                        T* d_out, 
                        NumItemsT num_items, 
                        int begin_bit,
                        int end_bit,
                        gamax::sort_order order,
                        void *d_temp_storage,
                        size_t temp_storage_bytes,
                        cudaStream_t stream) {
        bool alloc_temp_storage = (d_temp_storage == nullptr);
        std::optional<gamax::device_buffer> owned_temp_storage;
        if (alloc_temp_storage) {
            cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_in, d_out, num_items, begin_bit, end_bit, stream);
            owned_temp_storage.emplace(temp_storage_bytes, stream);
            d_temp_storage = owned_temp_storage->data();
        }
        if(order == gamax::sort_order::ASCENDING) {
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, begin_bit, end_bit, stream);
        }
        else {
            cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, begin_bit, end_bit, stream);
        }
    }

    template <typename KeyT, typename ValueT, typename NumItemsT = gamax::size_type>
    void radix_sort_kv(const KeyT* d_kin, 
                        KeyT* d_kout, 
                        const ValueT* d_vin,
                        ValueT* d_vout,
                        NumItemsT num_items, 
                        int begin_bit,
                        int end_bit,
                        gamax::sort_order order,
                        void *d_temp_storage,
                        size_t temp_storage_bytes,
                        cudaStream_t stream) {
        bool alloc_temp_storage = (d_temp_storage == nullptr);
        std::optional<gamax::device_buffer> owned_temp_storage;
        if (alloc_temp_storage) {
            cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_kin, d_kout, d_vin, d_vout, num_items, begin_bit, end_bit, stream);
            owned_temp_storage.emplace(temp_storage_bytes, stream);
            d_temp_storage = owned_temp_storage->data();
        }
        if(order == gamax::sort_order::ASCENDING) {
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_kin, d_kout, d_vin, d_vout, num_items, begin_bit, end_bit, stream);
        }
        else {
            cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_kin, d_kout, d_vin, d_vout, num_items, begin_bit, end_bit, stream);
        }
    }

    struct sort_key_value_helper {
        template <typename KeyT, typename ValueT, typename NumItemsT = gamax::size_type>
        void operator()(gamax::column_view d_kin, 
                        gamax::mutable_column_view d_kout, 
                        gamax::column_view d_vin,
                        gamax::mutable_column_view d_vout,
                        NumItemsT num_items, 
                        int begin_bit,
                        int end_bit,
                        gamax::sort_order order,
                        void *d_temp_storage,
                        size_t temp_storage_bytes,
                        cudaStream_t stream) {
            radix_sort_kv(d_kin.data<KeyT>(), d_kout.data<KeyT>(), d_vin.data<ValueT>(), d_vout.data<ValueT>(), num_items, begin_bit, end_bit, order, d_temp_storage, temp_storage_bytes, stream);
        }
    };

    struct sort_key_helper {
        template <typename T, typename NumItemsT = gamax::size_type>
        void operator()(gamax::column_view d_in, 
                        gamax::mutable_column_view d_out, 
                        NumItemsT num_items, 
                        int begin_bit,
                        int end_bit,
                        gamax::sort_order order,
                        void *d_temp_storage,
                        size_t temp_storage_bytes,
                        cudaStream_t stream) {
            radix_sort_k(d_in.data<T>(), d_out.data<T>(), num_items, begin_bit, end_bit, order, d_temp_storage, temp_storage_bytes, stream);
        }
    };
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
sort_column_pair(
    gamax::column& key_col, 
    gamax::column& val_col, 
    cudaStream_t stream,
    int begin_bit,
    int end_bit,
    gamax::sort_order order, 
    void* d_temp_storage, 
    size_t temp_storage_bytes) {
    
    return sort_column_pair(
        key_col.view(), val_col.view(), stream, begin_bit, end_bit, order,
        d_temp_storage, temp_storage_bytes);
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
sort_column_pair(
    gamax::column_view key_col, 
    gamax::column_view val_col, 
    cudaStream_t stream,
    int begin_bit,
    int end_bit,
    gamax::sort_order order, 
    void* d_temp_storage, 
    size_t temp_storage_bytes) {
    
    auto num_rows = key_col.size();

    auto key_out = std::make_unique<gamax::column>(key_col.type(), num_rows, stream);
    auto val_out = std::make_unique<gamax::column>(val_col.type(), num_rows, stream);

    if(end_bit <= 0) {
        end_bit = 8 * gamax::size_of(key_col.type());
    }

    double_type_dispatcher(
        key_col.type(), 
        val_col.type(), 
        sort_key_value_helper{}, 
        key_col, 
        key_out->mutable_view(), 
        val_col, 
        val_out->mutable_view(), 
        num_rows,
        begin_bit,
        end_bit,
        order, 
        d_temp_storage, 
        temp_storage_bytes, 
        stream
    );

    return std::make_pair(std::move(key_out), std::move(val_out));
}

std::unique_ptr<gamax::column>
sort_column(
    gamax::column& col,
    cudaStream_t stream,
    int begin_bit,
    int end_bit,
    gamax::sort_order order,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    return sort_column(
        col.view(), stream, begin_bit, end_bit, order,
        d_temp_storage, temp_storage_bytes);
}

std::unique_ptr<gamax::column>
sort_column(
    gamax::column_view col,
    cudaStream_t stream,
    int begin_bit,
    int end_bit,
    gamax::sort_order order,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
     auto num_rows = col.size();

    auto key_out = std::make_unique<gamax::column>(col.type(), num_rows, stream);

    if(end_bit <= 0) {
        end_bit = 8 * gamax::size_of(col.type());
    }

    type_dispatcher(
        col.type(), 
        sort_key_helper{}, 
        col, 
        key_out->mutable_view(), 
        num_rows,
        begin_bit,
        end_bit,
        order, 
        d_temp_storage, 
        temp_storage_bytes, 
        stream
    );

    return key_out;
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
argsort(
    gamax::column_view col,
    cudaStream_t stream,
    int begin_bit,
    int end_bit,
    gamax::sort_order order,
    void* d_temp_storage,
    size_t temp_storage_bytes) {
    
    auto idx_col = gamax::make_sequence(col.size(), stream);
    return sort_column_pair(
        col, idx_col->view(), stream, begin_bit, end_bit, order,
        d_temp_storage, temp_storage_bytes);
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
argsort(
    gamax::column& col,
    cudaStream_t stream,
    int begin_bit,
    int end_bit,
    gamax::sort_order order,
    void* d_temp_storage,
    size_t temp_storage_bytes) {
    
    return argsort(
        col.view(), stream, begin_bit, end_bit, order,
        d_temp_storage, temp_storage_bytes);
}

} // namespace gamax::sort
