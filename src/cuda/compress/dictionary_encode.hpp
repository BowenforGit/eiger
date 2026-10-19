// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <cuda.h>

#include "types/data_type.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"

namespace gamax::compress {

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::column_view column,
    cudaStream_t stream,
    const data_type encode_type = data_type{type_id::INT32},
    const gamax::size_type dictionary_size_hint = 0,
    const double load_factor = 0.25);

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode_with_order(
    const gamax::column_view column,
    cudaStream_t stream,
    const data_type encode_type = data_type{type_id::INT32},
    const gamax::size_type dictionary_size_hint = 0,
    const double load_factor = 0.25);

std::unique_ptr<gamax::column> dictionary_encode(
    const gamax::column_view column, 
    const gamax::column_view dict,
    cudaStream_t stream,
    const data_type encode_type = data_type{type_id::INT32});

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::table_view table,
    cudaStream_t stream,
    const data_type encode_type = data_type{type_id::INT32},
    const gamax::size_type dictionary_size_hint = 0,
    const double load_factor = 0.25);

} // namespace gamax::compress
