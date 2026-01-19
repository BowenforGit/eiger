#pragma once

#include "types/data_type.hpp"
#include "memory/table.hpp"

namespace gamax {
namespace groupby {

enum class aggregation_id {
    SUM,
    MAX,
    AVG,
    NOT_SUPPORTED_AGGREGATION
};

struct aggregation_request {
    gamax::size_type column_index;
    aggregation_id agg_kind;

    aggregation_request(gamax::size_type column_index, aggregation_id agg_kind)
        : column_index(column_index), agg_kind(agg_kind) {}
};

gamax::data_type agg_output_data_type(aggregation_id const& agg_kind, gamax::data_type const& type);
namespace aggregate {
std::unique_ptr<gamax::table> aggregate(
    gamax::table_view input, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    cudaStream_t stream = cudaStreamDefault);
}

#define DISPATCH_AGGREGATE(fn) \
template<typename... Ts> \
void dispatch_aggregate(gamax::groupby::aggregation_id agg_kind, Ts&&... args) { \
    switch(agg_kind) { \
        case gamax::groupby::aggregation_id::SUM: \
            fn<gamax::groupby::aggregation_id::SUM>{}(std::forward<Ts>(args)...); \
            break; \
        case gamax::groupby::aggregation_id::MAX: \
            fn<gamax::groupby::aggregation_id::MAX>{}(std::forward<Ts>(args)...); \
            break; \
        default: \
            throw std::runtime_error("Unsupported aggregation function"); \
    } \
}

} // namespace groupby
} // namespace gamax