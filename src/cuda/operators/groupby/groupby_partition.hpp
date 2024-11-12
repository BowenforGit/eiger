#pragma once

#define CUB_STDERR

#include <memory>
#include <vector>
#include <cuda.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/groupby.hpp"

namespace gamax::groupby::partition {
/**
 * @brief A partition-based groupby implementation.
 * 
 * @param key The key table.
 * @param val The value table.
 * @param agg_requests The aggregation requests. Each request corresponds to one aggregation in the final result
 * @param hint The groupby hint. Contains optional information such as the upper bound of the number of groups.
 * @param stream The CUDA stream to use.
 * 
 * @return A table containing the result of the groupby operation. The first few columns are the grouped keys and the rest is aggregations.
 * 
 * @throw if there are no keys or values. (neither pure grouping nor pure aggregating is supported)
 * @throw if the number of keys and values are not the same.
 * @throw if there are more than one keys. Currently we only support group by a single key
 * @throw if the key type and the value type are not int32_t or int64_t
 * @throw if there is no aggregation request (agg_requests is empty)
 * @throw if the aggregation type is not supported
 *
 * 
 * @note The input tables are not modified.
 * @note Currently, only max and sum are supported. 
 * 
 * Currently, the partition-based groupby has the following limitations.
 * 1. It cannot automatically choose the number of partitions to use. 
 * Currently, this number of partitions is fixed to be 2^16. 
 * We also do not have a good way to determine which radix bits should we use.
 * 2. The SMem-resident hash table size is limited and 
 * therefore cannot handle a large partition whose number of group may exceed the limit. 
 * This can lead to incorrect results. One solution to this is to use sort-based or hash-based groupby for those large partitions.
 * 3. When the partition is very large and for some reasons we are certain that the number of groups is small, 
 * we can still use the partition groupby. 
 * However, this will lead to a performance problem since we are only assigning one thread block to each partition.
 * 
 * @attention Only use the partition-based groupby when you are confident that all partitions have similar sizes and each partition has a small number of groups.
*/

// currently only sum and max are supported
std::unique_ptr<gamax::table> groupby(
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream = cudaStreamDefault);
} // namespace gamax::groupby::partition