#include "join.hpp"

#include <memory>
#include <vector>
#include <cuda.h>
#include <type_traits>

#include "materialize.hpp"
#include "merge_path.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/partition/partition.hpp"
#include "thrust/iterator/counting_iterator.h"

namespace gamax::join{
namespace detail {
using triple_col_t = std::tuple<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>>;
using tid_type = gamax::size_type;
constexpr 

triple_col_t partition_transform(
    gamax::column_view key,
    gamax::column_view val,
    int begin_bit,
    int end_bit,
    cudaStream_t stream)
{
    auto partitioned = gamax::partition::partition_column_pair(key, val, begin_bit, end_bit, nullptr, 0, stream);
    auto offset = gamax::partition::compute_partition_offsets(partitioned.first->view(), begin_bit, end_bit, stream);
    return std::make_tuple(std::move(partitioned.first), std::move(partitioned.second), std::move(offset));
} 


} // namespace detail

std::unique_ptr<gamax::table> ieq_partition_hash_join_gftr(
    gamax::table_view left, 
    gamax::table_view right,
    gamax::join::join_hint const& hint,
    cudaStream_t stream)
{
    // transformation
    auto [par_left_key, par_left_val] = partition_transform(left[0], left[1], stream);
    auto [par_right_key, par_right_val] = partition_transform(right[0], right[1], stream);
}
}

