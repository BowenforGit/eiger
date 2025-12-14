#pragma once

#include <cuda_runtime_api.h>

#include "string/string_column.hpp"
#include "memory/column.hpp"

namespace gamax {
column_ptr startswith(
    gamax::string_column_view in,
    gamax::string_scalar_view prefix,
    cudaStream_t stream = cudaStreamDefault
);

column_ptr endswith(
    gamax::string_column_view in,
    gamax::string_scalar_view suffix,
    cudaStream_t stream = cudaStreamDefault
);

column_ptr contains(
    gamax::string_column_view in,
    gamax::string_scalar_view substring,
    cudaStream_t stream = cudaStreamDefault
);
}