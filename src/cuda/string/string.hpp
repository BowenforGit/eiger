#pragma once

#include <cuda_runtime_api.h>

#include "string/string_column.hpp"
#include "memory/column.hpp"

namespace gamax {
struct string_op_options {
    int cg_size {1};
    int access_width {4};
    bool use_kmp {false};
    bool use_prefix_sort {false};
};

column_ptr equal(
    gamax::string_column_view in,
    gamax::string_scalar_view string,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr startswith(
    gamax::string_column_view in,
    gamax::string_scalar_view prefix,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr endswith(
    gamax::string_column_view in,
    gamax::string_scalar_view suffix,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr contains(
    gamax::string_column_view in,
    gamax::string_scalar_view substring,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr argsort_strings(
    gamax::string_column_view in,
    gamax::sort_order order,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);
}