// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "memory/column.hpp"
#include "memory/table.hpp"

namespace alps {
struct Context;
} // namespace alps

namespace gamax::tpch::dist {

std::unique_ptr<gamax::column> make_composite_hash_column(
    gamax::table_view keys,
    cudaStream_t stream);

std::vector<double> sum_doubles(
    const std::vector<double>& local,
    const alps::Context& ctx);

} // namespace gamax::tpch::dist
