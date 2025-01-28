#pragma once

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"

namespace gamax {
namespace util {
namespace device {
    /**
     * @brief Gather elements from input column based on gather_map
     * output[i] = input[gather_map[i]]
     */
    std::unique_ptr<gamax::column> gather(
        gamax::column_view gather_map, 
        gamax::column_view input, 
        cudaStream_t stream = cudaStreamDefault);

    std::unique_ptr<gamax::column> shuffle(
        gamax::column_view input,
        const int seed = 42,
        cudaStream_t stream = cudaStreamDefault
    );

    std::unique_ptr<gamax::column> compact( 
        gamax::column_view input, 
        gamax::column_view mask, 
        cudaStream_t stream = cudaStreamDefault);

    std::unique_ptr<gamax::column> compact( 
        gamax::column_view input, 
        gamax::column_view mask,
        gamax::size_type output_size,
        cudaStream_t stream = cudaStreamDefault);

    std::unique_ptr<gamax::table> compact(
        gamax::table_view input, 
        gamax::column_view mask, 
        cudaStream_t stream = cudaStreamDefault);
} // namespace device
} // namespace util
} // namespace gamax