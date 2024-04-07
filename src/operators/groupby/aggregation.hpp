#pragma once

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

} // namespace groupby
} // namespace gamax