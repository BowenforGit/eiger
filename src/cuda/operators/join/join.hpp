#pragma once

#include "types/data_type.hpp"
#include <memory>
#include <cuda.h>

#include "memory/column.hpp"
#include "memory/table.hpp"

namespace gamax {
    namespace join {
        constexpr gamax::size_type MAX_JOIN_OUTPUT_SIZE = (1 << 30); // max output size for join
        
        enum join_algo {
            AUTO_ALGO, // for runtime selection
            HASH,
            SORT,
            PARTITION,
            UNSUPPORTED_JOIN_ALGO
        };

        enum materialization_algo {
            AUTO_MAT, // for runtime selection
            GFUR,
            GFTR,
            INDEX_CLUSTER,
            UNSUPPORTED_MATERIALIZATION_ALGO
        };

        struct join_hint {
            bool sorted_left {false};
            bool sorted_right {false};
            bool unique_left {false};
            bool unique_right {false};
            gamax::size_type max_out {0}; // upper bound on the output size (0 means no limit)
            gamax::join::join_algo join_algo {gamax::join::AUTO_ALGO};
            gamax::join::materialization_algo mat_algo {gamax::join::AUTO_MAT};
        };

        std::unique_ptr<gamax::table> inner_eq_join(
            gamax::table_view left, 
            gamax::table_view right,
            gamax::join::join_hint const& hint,
            cudaStream_t stream = cudaStreamDefault);
        
        namespace detail {
            std::unique_ptr<gamax::table> ieq_hash_join(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream = cudaStreamDefault);
            
            std::unique_ptr<gamax::table> ieq_sort_merge_join_gfur(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream = cudaStreamDefault);
            
            std::unique_ptr<gamax::table> ieq_sort_merge_join_gftr(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream = cudaStreamDefault);
            
            std::unique_ptr<gamax::table> ieq_partition_hash_join_gftr(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream = cudaStreamDefault);

            std::unique_ptr<gamax::table> ieq_partition_hash_join_gfur(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream = cudaStreamDefault);
            
            gamax::size_type get_join_output_alloc_size(
                gamax::table_view const& left, 
                gamax::table_view const& right,
                gamax::join::join_hint const& hint);
        } // namespace detail
    } // namespace join
} // namespace gamax