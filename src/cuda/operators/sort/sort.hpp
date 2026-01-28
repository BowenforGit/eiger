#pragma once

#define CUB_STDERR

#include <cuda_runtime_api.h> // cudaStream_t
#include <memory> // std::unique_ptr

#include "types/data_type.hpp" // sort_order
#include "memory/column.hpp" // column
#include "memory/table.hpp" // table

namespace gamax {
namespace sort {
std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
sort_column_pair(
    gamax::column& key_col, 
    gamax::column& val_col, 
    int begin_bit = 0,
    int end_bit = -1,
    gamax::sort_order order = gamax::sort_order::ASCENDING, 
    void* d_temp_storage = nullptr, 
    size_t temp_storage_bytes = 0, 
    cudaStream_t stream = 0);

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
sort_column_pair(
    gamax::column_view key_col, 
    gamax::column_view val_col, 
    int begin_bit = 0,
    int end_bit = -1,
    gamax::sort_order order = gamax::sort_order::ASCENDING, 
    void* d_temp_storage = nullptr, 
    size_t temp_storage_bytes = 0, 
    cudaStream_t stream = 0);

std::unique_ptr<gamax::column>
sort_column(
    gamax::column& col,
    int begin_bit = 0,
    int end_bit = -1,
    gamax::sort_order order = gamax::sort_order::ASCENDING,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0,
    cudaStream_t stream = 0);

std::unique_ptr<gamax::column>
sort_column(
    gamax::column_view col,
    int begin_bit = 0,
    int end_bit = -1,
    gamax::sort_order order = gamax::sort_order::ASCENDING,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0,
    cudaStream_t stream = 0);


std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
argsort(
    gamax::column& col,
    int begin_bit = 0,
    int end_bit = -1,
    gamax::sort_order order = gamax::sort_order::ASCENDING,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0,
    cudaStream_t stream = 0);

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
argsort(
    gamax::column_view col,
    int begin_bit = 0,
    int end_bit = -1,
    gamax::sort_order order = gamax::sort_order::ASCENDING,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0,
    cudaStream_t stream = 0);


struct sort_table_options {
    bool smart_keys_fusion {false};
};

std::unique_ptr<gamax::table>
sort_table(
    gamax::table_view const& table,
    std::vector<int> const& sort_keys,
    std::vector<gamax::sort_order> const& sort_orders,
    cudaStream_t stream = 0);

std::unique_ptr<gamax::table>
sort_table(
    gamax::table_view const& table,
    std::vector<int> const& sort_keys,
    std::vector<gamax::sort_order> const& sort_orders,
    sort_table_options const& options,
    cudaStream_t stream = 0);

std::unique_ptr<gamax::table>
sort_table(
    gamax::table_view const& table,
    int sort_key_index,
    gamax::sort_order sort_order,
    cudaStream_t stream = 0);

std::unique_ptr<gamax::table>
topk_table(
    gamax::table_view const& table,
    std::vector<int> const& sort_keys,
    std::vector<gamax::sort_order> const& sort_orders,
    gamax::size_type k,
    bool result_in_sorted_order,
    cudaStream_t stream = 0);
} // namespace sort
} // namespace gamax