#pragma once

#define CUB_STDERR

#include <memory>

#include <cuda.h>
#include <cub/cub.cuh> 

// #include "types/tuple.cuh"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "memory/column.hpp"
#include "utils/utils.cuh"
#include "utils/sequence.cuh"

namespace gamax {
namespace sort {
std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
sort_column_pair(
    gamax::column& key_col, 
    gamax::column& val_col, 
    gamax::sort_order order = gamax::sort_order::ASCENDING, 
    void* d_temp_storage = nullptr, 
    size_t temp_storage_bytes = 0, 
    cudaStream_t stream = 0);

std::unique_ptr<gamax::column>
sort_column(
    gamax::column& col,
    gamax::sort_order order = gamax::sort_order::ASCENDING,
    void* d_temp_storage = nullptr,
    size_t temp_storage_bytes = 0,
    cudaStream_t stream = 0);
} // namespace sort
} // namespace gamax