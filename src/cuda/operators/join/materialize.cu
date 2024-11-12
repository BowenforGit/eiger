#include "materialize.hpp"

#include <memory>
#include <cuda.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
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
        return gamax::util::device::gather(map, col, stream);
    }
} // detail
} // join
} // gamax