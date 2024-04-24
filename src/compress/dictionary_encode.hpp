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
    cudaStream_t stream = 0);

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::table& table, 
    const gamax::size_type dictionary_size_hint = 0,
    const double load_factor = 0.75,
    cudaStream_t stream = 0);

} // namespace gamax::compress