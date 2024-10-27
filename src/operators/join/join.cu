#include "join.hpp"

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"

namespace gamax {
namespace join {
namespace {
void check_join_solvable(
    gamax::table_view left, 
    gamax::table_view right,
    gamax::join::join_hint const& hint) 
{
    if(hint.join_algo == gamax::join::AUTO_ALGO || hint.mat_algo == gamax::join::AUTO_MAT) {
        throw std::runtime_error("Join algorithm and materialization algorithm do not support auto selection");
    }

    if(hint.max_out > gamax::join::MAX_JOIN_OUTPUT_SIZE) {
        throw std::runtime_error("Output size exceeds the maximum limit");
    }

    if(hint.mat_algo == gamax::join::INDEX_CLUSTER) {
        throw std::runtime_error("Index cluster materialization algorithm is not supported");
    }
}
}

std::unique_ptr<gamax::table> inner_eq_join(
    gamax::table_view left, 
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream) 
{
    check_join_solvable(left, right, hint);
    
    switch (hint.join_algo) {
        case gamax::join::SORT:
            // return hint.mat_algo == gamax::join::GFUR ?
            //     detail::ieq_sort_merge_join_gfur(left, right, hint, stream) :
            //     detail::ieq_sort_merge_join_gftr(left, right, hint, stream);
            return detail::ieq_sort_merge_join_gftr(left, right, hint, stream);
        case gamax::join::HASH:
            // return detail::ieq_hash_join(left, right, hint, stream);
        case gamax::join::PARTITION:
            // return hint.mat_algo == gamax::join::GFUR ?
            //     detail::ieq_partition_hash_join_gfur(left, right, hint, stream) :
            //     detail::ieq_partition_hash_join_gftr(left, right, hint, stream);
        default:
            throw std::runtime_error("Unsupported join algorithm");
    }

}

namespace detail {
gamax::size_type get_join_output_alloc_size(
    gamax::table_view const& left, 
    gamax::table_view const& right,
    gamax::join::join_hint const& hint) {

    if(hint.max_out != -1) return hint.max_out;
    return std::min(
        hint.unique_left ? right.num_rows() : gamax::join::MAX_JOIN_OUTPUT_SIZE,
        hint.unique_right ? left.num_rows() : gamax::join::MAX_JOIN_OUTPUT_SIZE);
}
} // namespace detail
} // namespace join
} // namespace gamax