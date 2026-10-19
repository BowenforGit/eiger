// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "materialize.hpp"

#include <memory>
#include <cuda.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"
#include "utils/transform.hpp"


namespace gamax {
namespace join {
namespace detail {
    std::unique_ptr<gamax::column> simple_materialize(
        gamax::column_view col,
        gamax::column_view map,
        cudaStream_t stream)
    {
        // util::device::gather's internal double_type_dispatcher doesn't
        // admit STRING. Dispatch here so every caller (hash_join,
        // sort_merge_join, cross_join, groupby's leader-tid gather)
        // transparently handles both numeric and STRING columns.
        if (col.type().id() == gamax::type_id::STRING) {
            gamax::string_column_view sv(col, gamax::ARROW_STRING);
            auto gathered = gamax::gather_strings(sv, map, stream);
            return std::unique_ptr<gamax::column>(std::move(gathered));
        }
        return gamax::util::device::gather(map, col, stream);
    }
} // detail
} // join
} // gamax
