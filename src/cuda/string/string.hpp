#pragma once

#include <cuda_runtime_api.h>
#include <vector>

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

column_ptr equal_bitmask(
    gamax::string_column_view in,
    gamax::string_scalar_view string,
    gamax::column_view const& mask,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr startswith(
    gamax::string_column_view in,
    gamax::string_scalar_view prefix,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr startswith_bitmask(
    gamax::string_column_view in,
    gamax::string_scalar_view prefix,
    gamax::column_view const& mask,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr endswith(
    gamax::string_column_view in,
    gamax::string_scalar_view suffix,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr endswith_bitmask(
    gamax::string_column_view in,
    gamax::string_scalar_view suffix,
    gamax::column_view const& mask,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr contains(
    gamax::string_column_view in,
    gamax::string_scalar_view substring,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr contains_bitmask(
    gamax::string_column_view in,
    gamax::string_scalar_view substring,
    gamax::column_view const& mask,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr contains_multiple(
    gamax::string_column_view in,
    const std::vector<gamax::string_scalar_view>& patterns,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr contains_multiple_bitmask(
    gamax::string_column_view in,
    const std::vector<gamax::string_scalar_view>& patterns,
    gamax::column_view const& mask,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

// `x IN (p0, p1, ...)` — true iff `x` exactly equals any of the
// supplied patterns. Mirrors `contains_multiple`'s interface and
// shares its shared-memory pattern staging (`multi_pattern_helper`
// in string_fast.cu); the inner kernel loop is a length-pre-check +
// full-string compare, short-circuiting on the first match.
column_ptr in_multiple(
    gamax::string_column_view in,
    const std::vector<gamax::string_scalar_view>& patterns,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

column_ptr in_multiple_bitmask(
    gamax::string_column_view in,
    const std::vector<gamax::string_scalar_view>& patterns,
    gamax::column_view const& mask,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

// SQL SUBSTRING(str FROM offset FOR length). `sql_offset` is 1-based,
// matching the SQL convention the IR builders hand us (e.g.
// `substring(c_phone, 1, 2)` takes the first two characters). Rows
// whose effective start is past the end of the input produce an
// empty string. `length` clamps to the remaining row width.
string_column_ptr substring(
    gamax::string_column_view in,
    gamax::size_type sql_offset,
    gamax::size_type length,
    cudaStream_t stream = cudaStreamDefault);

column_ptr argsort_strings(
    gamax::string_column_view in,
    gamax::sort_order order,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

string_column_ptr sort_strings(
    gamax::string_column_view in,
    gamax::sort_order order,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);

string_column_ptr gather_strings(
    gamax::string_column_view in,
    gamax::column_view const& gather_map,
    cudaStream_t stream = cudaStreamDefault,
    string_op_options options = {}
);
}