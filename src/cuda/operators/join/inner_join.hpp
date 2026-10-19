// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <functional>
#include <memory>
#include <cuda_runtime_api.h>

#include "memory/table.hpp"
#include "operators/join/join.hpp"

namespace gamax {
    namespace join {
        using BeforeProbeHook = std::function<void(cudaStream_t)>;

        struct EqJoinMatches {
            std::unique_ptr<gamax::column> left_indices;
            std::unique_ptr<gamax::column> right_indices;
        };

        // HASH equality match without payload materialization. The returned
        // columns contain corresponding left/build and right/probe row indices.
        // This intentionally retains the HASH implementation's current host
        // synchronization while resolving the actual match count.
        EqJoinMatches eq_join_indices(
            gamax::column_view left_key,
            gamax::column_view right_key,
            gamax::join::join_hint const& hint,
            cudaStream_t stream);

        EqJoinMatches eq_join_indices(
            gamax::column_view left_key,
            gamax::column_view right_key,
            gamax::join::join_hint const& hint,
            cudaStream_t stream,
            const BeforeProbeHook& before_probe);

        std::unique_ptr<gamax::table> inner_eq_join(
            gamax::table_view left, 
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream);
        
        namespace detail {
            std::unique_ptr<gamax::table> ieq_hash_join(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream);
            
            std::unique_ptr<gamax::table> ieq_sort_merge_join_gfur(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream);
            
            std::unique_ptr<gamax::table> ieq_sort_merge_join_gftr(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream);
            
            std::unique_ptr<gamax::table> ieq_partition_hash_join_gftr(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream);

            std::unique_ptr<gamax::table> ieq_partition_hash_join_gfur(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream);
            
            gamax::size_type get_ieq_join_output_alloc_size(
                gamax::table_view const& left, 
                gamax::table_view const& right,
                gamax::join::join_hint const& hint);
        } // namespace detail
    } // namespace join
} // namespace gamax
