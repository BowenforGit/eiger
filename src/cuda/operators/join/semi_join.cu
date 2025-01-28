#include "semi_join.hpp"

#include <cuda.h>
#include <memory>
#include <cuda.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/join.hpp"

namespace gamax {
namespace join {
namespace {
void check_join_solvable(
    gamax::table_view left, 
    gamax::table_view right,
    gamax::join::join_hint const& hint) 
{
    if(hint.join_algo == gamax::join::AUTO_ALGO || hint.mat_algo == gamax::join::AUTO_MAT) {
        throw std::runtime_error("Semi-join algorithm and materialization algorithm do not support auto selection");
    }

    if(hint.max_out > gamax::join::MAX_JOIN_OUTPUT_SIZE) {
        throw std::runtime_error("[semi-join] Output size exceeds the maximum limit");
    }

    if(hint.mat_algo == gamax::join::INDEX_CLUSTER) {
        throw std::runtime_error("[semi-join] Index cluster materialization algorithm is not supported");
    }

    if(left[0].type() != right[0].type()) {
        throw std::runtime_error("[semi-join] Join columns cannot have different types");
    }

    if (left.num_columns() == 0 || right.num_columns() == 0) {
        throw std::runtime_error("[semi-join] Empty table");
    }
}
}

namespace detail {
    gamax::size_type get_semi_join_output_alloc_size(
        gamax::table_view const& left, 
        gamax::table_view const& right,
        gamax::join::join_hint const& hint) {

        if(hint.max_out != 0) return std::min(hint.max_out, left.num_rows());
        return left.num_rows();
    }
} // namespace detail

std::unique_ptr<gamax::table> semi_join(
    gamax::table_view left, 
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream) 
{
    check_join_solvable(left, right, hint);
    
    switch (hint.join_algo) {
        case gamax::join::HASH:
            return detail::semi_hash_join(left, right, hint, stream);
        case gamax::join::PARTITION:
            // return hint.mat_algo == gamax::join::GFUR ?
            //     detail::semi_partition_hash_join_gfur(left, right, hint, stream) :
            //     detail::semi_partition_hash_join_gftr(left, right, hint, stream);
        default:
            throw std::runtime_error("Unsupported semi-join algorithm");
    }
}
} // namespace join
} // namespace gamax