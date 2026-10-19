// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <cuda_runtime_api.h>

#include "memory/column.hpp"

namespace gamax {

/**
 * @brief Create a deep copy of a column from a column_view.
 * 
 * Performs a device-to-device memory copy of the column data.
 *
 * @param view The source column view to copy from
 * @param stream CUDA stream for async execution
 * @return New column with copied data
 */
std::unique_ptr<column> copy_column(
    column_view const& view,
    cudaStream_t stream);

} // namespace gamax
