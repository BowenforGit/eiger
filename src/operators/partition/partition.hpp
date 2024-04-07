#pragma once

#define CUB_STDERR

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"

namespace gamax::partition {
std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
partition_column_pair(
    gamax::column& key_col, 
    gamax::column& val_col, 
    int begin_bit,
    int end_bit,
    void* d_temp_storage = nullptr, 
    size_t temp_storage_bytes = 0, 
    cudaStream_t stream = 0);

std::unique_ptr<gamax::column>
partition_column(
    gamax::column& col,
    int begin_bit,
    int end_bit,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0,
    cudaStream_t stream = 0);

std::unique_ptr<gamax::column>
compute_partition_offsets(
    gamax::column& partitioned_key_col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream = 0);
} // namespace gamax::partition