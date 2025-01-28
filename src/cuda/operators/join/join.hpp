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
    } // namespace join
} // namespace gamax