#pragma once

#define CUB_STDERR

#include <cuda_runtime_api.h> // cudaStream_t
#include <memory> // std::unique_ptr

#include "types/data_type.hpp" // sort_order
#include "memory/column.hpp" // column

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
} // namespace sort
} // namespace gamax