#pragma once

#include "types/data_type.hpp"
#include <memory>
#include <cuda.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/join.hpp"

namespace gamax {
    namespace join {
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
            
            gamax::size_type get_ieq_join_output_alloc_size(
                gamax::table_view const& left, 
                gamax::table_view const& right,
                gamax::join::join_hint const& hint);
        } // namespace detail
    } // namespace join
} // namespace gamax