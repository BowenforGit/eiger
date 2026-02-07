#pragma once

#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"

namespace gamax::compress {
    struct multikey_compress_options {
        // if true, always try to compress even if there's only one key column
        bool always_compress = true;

        // Maximum total bits that can be fused into a single uint64_t key.
        // If the required bits exceed this limit, compression fails (returns nullptr).
        int max_fused_key_bits = 64;

        // If non-null, the actual number of fused bits used is written here.
        // Useful for callers (e.g. radix sort) that benefit from knowing end_bit.
        int* out_total_bits = nullptr;
    };

    // Generic multi-key compression (equality-preserving).
    // Uses non-order-preserving dictionary encoding when beneficial.
    // All columns in the table are treated as keys.
    // Returns nullptr if compression is not possible.
    column_ptr multikey_compress(
        table_view table, 
        multikey_compress_options const& options = {},
        cudaStream_t stream = cudaStreamDefault);
    
    // Sort-aware multi-key compression (order-preserving).
    // Uses order-preserving dictionary encoding and respects sort orders
    // (descending keys have their bits flipped for correct radix sort behavior).
    // Returns nullptr if compression is not possible.
    column_ptr multikey_compress(
        table_view table, 
        std::vector<gamax::sort_order> const& sort_orders,
        multikey_compress_options const& options = {},
        cudaStream_t stream = cudaStreamDefault);
};