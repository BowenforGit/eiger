#include "groupby.hpp"

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/groupby_hash.hpp"
#include "operators/groupby/groupby_partition.hpp"

namespace gamax::groupby {
    std::unique_ptr<gamax::table> groupby(
        gamax::table_view key, 
        gamax::table_view val, 
        std::vector<gamax::groupby::aggregation_request>& agg_requests,
        gamax::groupby::groupby_hint& hint,
        cudaStream_t stream)
    {   
        switch (hint.groupby_algo) {
            case gamax::groupby::AUTO_ALGO:
            case gamax::groupby::SORT:
                return gamax::groupby::sort::groupby(key, val, agg_requests, hint, stream);
            case gamax::groupby::SORT_GFUR:
                return gamax::groupby::sort::groupby_gfur(key, val, agg_requests, hint, stream);
            case gamax::groupby::HASH:
                return gamax::groupby::hash::groupby(key, val, agg_requests, hint, stream);
            // case gamax::groupby::PARTITION:
            //     return gamax::groupby::partition::groupby(key, val, agg_requests, hint, stream);
            // case gamax::groupby::PARTITION_GFUR:
            //     return gamax::groupby::partition::groupby_gfur(key, val, agg_requests, hint, stream);
            default:
                throw std::runtime_error("Unsupported groupby algorithm");
        }
    }
} // namespace gamax::groupby