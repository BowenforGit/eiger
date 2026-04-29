#include "aggregation.hpp"

#include <cmath>
#include <cuda.h>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "types/data_type.hpp"

namespace gamax::groupby {
namespace {
struct agg_output_data_type_helper {
    template <typename T>
    gamax::data_type operator()(aggregation_id const& agg_kind) const {
        switch(agg_kind) {
            case gamax::groupby::aggregation_id::SUM:
                return gamax::data_type{gamax::type_to_id<typename gamax::groupby::Sum<T>::OutputType>()};
            case gamax::groupby::aggregation_id::MAX:
                return gamax::data_type{gamax::type_to_id<typename gamax::groupby::Max<T>::OutputType>()};
            case gamax::groupby::aggregation_id::MIN:
                return gamax::data_type{gamax::type_to_id<typename gamax::groupby::Min<T>::OutputType>()};
            case gamax::groupby::aggregation_id::ANY:
                return gamax::data_type{gamax::type_to_id<typename gamax::groupby::Any<T>::OutputType>()};
            case gamax::groupby::aggregation_id::AVG:
                return gamax::data_type{gamax::type_id::FLOAT64}; // AVG always returns double
            default:
                throw std::runtime_error("Unsupported aggregation function");
        }
    }
};
}

gamax::data_type agg_output_data_type(aggregation_id const& agg_kind, gamax::data_type const& type) {
    // COUNT_ALL's output is INT64 regardless of input type — it
    // doesn't read an input column at all. Short-circuit before the
    // type_dispatcher to avoid forcing callers to supply a valid
    // `type` for this kind.
    if (agg_kind == aggregation_id::COUNT_ALL) {
        return gamax::data_type{gamax::type_id::INT64};
    }
    return type_dispatcher(
        type,
        agg_output_data_type_helper{},
        agg_kind);
}

namespace aggregate {

namespace {
template<gamax::groupby::aggregation_id agg_kind>
struct reduction_helper {
    template<typename T>
    void operator() (
        gamax::column_view val,
        gamax::mutable_column_view result_val,
        cudaStream_t stream)
    {
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;
        auto res = thrust::reduce(thrust::cuda::par(*get_thrust_allocator()).on(stream), val.data<T>(), val.data<T>() + val.size(), F::identity, F{});
        cudaMemcpyAsync(result_val.data<typename F::OutputType>(), &res, sizeof(res), cudaMemcpyHostToDevice, stream);
    }
};

template<gamax::groupby::aggregation_id agg_kind>
struct dispatch_val_type_helper {
    void operator() (
        gamax::column_view val,
        gamax::mutable_column_view result_val,
        cudaStream_t stream) {

        gamax::type_dispatcher(
            val.type(),
            reduction_helper<agg_kind>{},
            val,
            result_val,
            stream
        );
    }
};

DISPATCH_AGGREGATE(dispatch_val_type_helper, __host__)

// Scalar AVG: compute SUM in the SUM output-type space, then divide by
// the (known) row count on host and copy a single FLOAT64 cell into the
// result column. Count is exact because it's just `val.size()`. The
// grouped path in `groupby.cu` derives count from a SUM-of-ones kernel
// (it doesn't know per-group cardinality up front); we don't need that
// here.
struct avg_reduction_helper {
    template<typename T>
    void operator() (
        gamax::column_view val,
        gamax::mutable_column_view result_val,
        cudaStream_t stream)
    {
        using SumF = typename gamax::groupby::Sum<T>;
        using SumT = typename SumF::OutputType;
        auto sum = thrust::reduce(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            val.data<T>(), val.data<T>() + val.size(),
            SumF::identity, SumF{});
        cudaStreamSynchronize(stream);
        double avg = static_cast<double>(sum) /
                     static_cast<double>(val.size());
        cudaMemcpyAsync(result_val.data<double>(), &avg, sizeof(double),
                        cudaMemcpyHostToDevice, stream);
    }
};

void dispatch_avg(
    gamax::column_view val,
    gamax::mutable_column_view result_val,
    cudaStream_t stream)
{
    gamax::type_dispatcher(val.type(), avg_reduction_helper{},
                           val, result_val, stream);
}

}



std::unique_ptr<gamax::table> aggregate(
    gamax::table_view input,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    cudaStream_t stream)
{
    std::vector<std::unique_ptr<gamax::column>> final_columns;

    for (auto const& ar : agg_requests) {
        auto agg_kind = ar.agg_kind;
        // COUNT_ALL doesn't read an input column — the result is the
        // table's row count emitted as a single INT64. Handle it
        // before touching `input[ar.column_index]` because callers
        // pass column_index=-1 by convention.
        if (agg_kind == gamax::groupby::aggregation_id::COUNT_ALL) {
            auto res_col = gamax::make_column<int64_t>(1, false, stream);
            int64_t n = static_cast<int64_t>(input.num_rows());
            cudaMemcpyAsync(res_col->data<int64_t>(), &n, sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream);
            final_columns.push_back(std::move(res_col));
            continue;
        }
        auto col = input[ar.column_index];
        auto res_col = gamax::make_column(agg_output_data_type(agg_kind, col.type()), 1);
        if (agg_kind == gamax::groupby::aggregation_id::AVG) {
            dispatch_avg(col, res_col->mutable_view(), stream);
        } else {
            dispatch_aggregate(agg_kind, col, res_col->mutable_view(), stream);
        }
        final_columns.push_back(std::move(res_col));
    }

    cudaStreamSynchronize(stream);

    return std::make_unique<gamax::table>(std::move(final_columns));
}
} // namespace aggregate
} // namespace gamax::groupby
