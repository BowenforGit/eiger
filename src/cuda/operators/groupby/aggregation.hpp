#pragma once

#include "types/data_type.hpp"
#include "agg_func.cuh"
#include "memory/table.hpp"

namespace gamax {
namespace groupby {

enum class aggregation_id {
    SUM,
    MAX,
    NOT_SUPPORTED_AGGREGATION
};

struct aggregation_request {
    int column_index;
    aggregation_id agg_kind;

    aggregation_request(int column_index, aggregation_id agg_kind)
        : column_index(column_index), agg_kind(agg_kind) {}
};

struct agg_output_data_type_helper {
    template <typename T>
    gamax::data_type operator()(aggregation_id const& agg_kind) const {
        switch(agg_kind) {
            case gamax::groupby::aggregation_id::SUM:
                return gamax::data_type{gamax::type_to_id<typename gamax::groupby::Sum<T>::OutputType>()};
            case gamax::groupby::aggregation_id::MAX:
                return gamax::data_type{gamax::type_to_id<typename gamax::groupby::Sum<T>::OutputType>()};
            default:
                throw std::runtime_error("Unsupported aggregation function");
        }
    }
};

__inline__ gamax::data_type agg_output_data_type(aggregation_id const& agg_kind, gamax::data_type const& type) {
    return type_dispatcher(
        type,
        agg_output_data_type_helper{},
        agg_kind);
}

template<gamax::groupby::aggregation_id agg_kind, typename ValT>
struct agg_fn_mapper {
    using F = void;
};

#define AGG_FN_MAPPER(agg_kind, agg_name) \
    template<typename ValT> \
    struct agg_fn_mapper<gamax::groupby::aggregation_id::agg_kind, ValT> { \
        using F = gamax::groupby::agg_name<ValT>; \
    };

AGG_FN_MAPPER(SUM, Sum)
AGG_FN_MAPPER(MAX, Max)

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