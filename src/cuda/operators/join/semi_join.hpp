// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <cuda_runtime_api.h>

#include "memory/table.hpp"
#include "operators/join/join.hpp"

namespace gamax {
    namespace join {
        // return tuples from left table that have a match in the right table
        std::unique_ptr<gamax::table> semi_join(
            gamax::table_view left, 
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream);
        
        namespace detail {
            // Type-erased build-side HASH set. Splitting build from probe lets
            // distributed schedules construct the right-key set while probe
            // columns are still being exchanged.
            class semi_hash_set {
            public:
                virtual ~semi_hash_set() = default;
                virtual std::unique_ptr<gamax::column> contains(
                    gamax::column_view keys,
                    cudaStream_t stream) = 0;
            };

            std::unique_ptr<semi_hash_set> build_semi_hash_set(
                gamax::column_view right_key,
                cudaStream_t stream);

            // Compute the per-left-row membership bitmask for an equi
            // semi-join on the single-column key pair (left[0], right[0]):
            // result[i] == true iff `left[0][i]` is found among right's
            // keys. Exposed so `anti_join` can reuse the same hash-set
            // build + contains pass and then negate the mask instead
            // of compacting against it directly.
            std::unique_ptr<gamax::column> semi_hash_membership_mask(
                gamax::table_view left,
                gamax::table_view right,
                cudaStream_t stream);

            std::unique_ptr<gamax::table> semi_hash_join(
                gamax::table_view left,
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream);
            
            std::unique_ptr<gamax::table> semi_partition_hash_join_gftr(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream);

            std::unique_ptr<gamax::table> semi_partition_hash_join_gfur(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream);
            
            gamax::size_type get_semi_join_output_alloc_size(
                gamax::table_view const& left, 
                gamax::table_view const& right,
                gamax::join::join_hint const& hint);
        } // namespace detail
    } // namespace join
} // namespace gamax
