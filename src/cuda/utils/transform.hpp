// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"

namespace gamax {

enum prefix_sum_options {
    INCLUSIVE,
    EXCLUSIVE,
    INCLUSIVE_PAD_ZERO
};

namespace util {
namespace device {
    /**
     * @brief Gather into a caller-owned output view.
     *
     * This is the allocation-free counterpart of gather(). The output must
     * have the same type as input and contain at least gather_map.size()
     * elements.
     */
    void gather_into(
        gamax::column_view gather_map,
        gamax::column_view input,
        gamax::mutable_column_view output,
        cudaStream_t stream);

    /**
     * @brief Gather elements from input column based on gather_map
     * output[i] = input[gather_map[i]]
     */
    std::unique_ptr<gamax::column> gather(
        gamax::column_view gather_map, 
        gamax::column_view input, 
        cudaStream_t stream);

    std::unique_ptr<gamax::column> shuffle(
        gamax::column_view input,
        cudaStream_t stream,
        const int seed = 42
    );

    std::unique_ptr<gamax::column> compact( 
        gamax::column_view input, 
        gamax::column_view mask, 
        cudaStream_t stream);

    std::unique_ptr<gamax::column> compact( 
        gamax::column_view input, 
        gamax::column_view mask,
        gamax::size_type output_size,
        cudaStream_t stream);

    std::unique_ptr<gamax::table> compact(
        gamax::table_view input, 
        gamax::column_view mask, 
        cudaStream_t stream);

    std::unique_ptr<gamax::column> prefix_sum( 
        gamax::column_view input, 
        gamax::prefix_sum_options option,
        gamax::data_type output_type,
        cudaStream_t stream);
    
    gamax::size_type count_nonzero(
        gamax::column_view input,
        cudaStream_t stream);

    std::unique_ptr<gamax::column> bitmask_and(
        gamax::column_view lhs,
        gamax::column_view rhs,
        cudaStream_t stream);

    std::unique_ptr<gamax::column> bitmask_or(
        gamax::column_view lhs,
        gamax::column_view rhs,
        cudaStream_t stream);

    std::unique_ptr<gamax::column> bitmask_not(
        gamax::column_view input,
        cudaStream_t stream);
} // namespace device
} // namespace util
} // namespace gamax
