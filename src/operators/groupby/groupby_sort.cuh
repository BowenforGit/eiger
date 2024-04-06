#pragma once

#define CUB_STDERR

#include <iostream>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "utils/utils.cuh"
#include "utils/cub_wrapper.cuh"
#include "operators/sort/sort.cuh"
#include "operators/groupby/load_balancer.cuh"
#include "operators/groupby/aggregation.hpp"

namespace gamax::groupby::sort {

// currently only sum and max are supported
std::unique_ptr<gamax::table> groupby(
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint);
    
} // namespace gamax::groupby::sort