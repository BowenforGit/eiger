#pragma once

#include <memory>
#include <cuda_runtime_api.h>
#include <functional>

#include "memory/column.hpp"

namespace gamax {
namespace join {
namespace detail {
    std::unique_ptr<gamax::column> simple_materialize(
        gamax::column_view col,
        gamax::column_view map,
        cudaStream_t stream = cudaStreamDefault);
} // namespace detail
} // namespace join
} // namespace gamax