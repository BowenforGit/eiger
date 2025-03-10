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
            cudaStream_t stream = cudaStreamDefault);
        
        namespace detail {
            std::unique_ptr<gamax::table> semi_hash_join(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream = cudaStreamDefault);
            
            std::unique_ptr<gamax::table> semi_partition_hash_join_gftr(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream = cudaStreamDefault);

            std::unique_ptr<gamax::table> semi_partition_hash_join_gfur(
                gamax::table_view left, 
                gamax::table_view right,
                gamax::join::join_hint const& hint,
                cudaStream_t stream = cudaStreamDefault);
            
            gamax::size_type get_semi_join_output_alloc_size(
                gamax::table_view const& left, 
                gamax::table_view const& right,
                gamax::join::join_hint const& hint);
        } // namespace detail
    } // namespace join
} // namespace gamax