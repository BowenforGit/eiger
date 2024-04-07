#pragma once

#include "types/data_type.hpp"

namespace gamax {
namespace groupby {
struct groupby_hint {
    gamax::size_type groups_upper_bound {gamax::MAXIMUM_SIZE};
};
}
}