#pragma once

#include <memory>
#include <cuda.h>

#include "types/data_type.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"

namespace gamax::compress {

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::column& column, 
    const gamax::size_type dictionary_size_hint = 0,
    const double load_factor = 0.75,
    cudaStream_t stream = cudaStreamDefault);

std::unique_ptr<gamax::column> dictionary_encode(
    const gamax::column& column, 
    const gamax::column& dict,
    const data_type encode_type = data_type{type_id::INT32},
    cudaStream_t stream = cudaStreamDefault);

std::unique_ptr<gamax::column> dictionary_encode(
    const gamax::column_view column, 
    const gamax::column_view dict,
    const data_type encode_type = data_type{type_id::INT32},
    cudaStream_t stream = cudaStreamDefault);

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::table& table, 
    const gamax::size_type dictionary_size_hint = 0,
    const double load_factor = 0.75,
    cudaStream_t stream = cudaStreamDefault);

} // namespace gamax::compress