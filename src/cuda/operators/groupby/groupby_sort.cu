#include <memory>
#include <type_traits>
#include <thrust/iterator/permutation_iterator.h>

#include "utils/utils.cuh"
#include "utils/cub_wrapper.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "operators/sort/sort.hpp"
#include "types/data_type.hpp"
#include "compress/dictionary_encode.hpp"

namespace gamax::groupby::sort {

namespace {

template<typename F, typename ValT>
struct reduce_by_key_aggregator_helper {
    template<typename KeyT>
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        gamax::mutable_column_view result_key,
        gamax::mutable_column_view result_val,
        gamax::size_type* d_num_groups,
        gamax::size_type* permutation_idx = nullptr) {

        using AggT = typename F::OutputType;
        
        auto num_items = key.size();
        
        if(permutation_idx != nullptr) {
            auto val_itr = thrust::make_permutation_iterator(val.data<ValT>(), permutation_idx);
            CALL_CUB_FUNC(cub::DeviceReduce::ReduceByKey, key.data<KeyT>(), result_key.data<KeyT>(), val_itr, result_val.data<AggT>(), d_num_groups, F{}, num_items)
        } else {
            CALL_CUB_FUNC(cub::DeviceReduce::ReduceByKey, key.data<KeyT>(), result_key.data<KeyT>(), val.data<ValT>(), result_val.data<AggT>(), d_num_groups, F{}, num_items)
        }
    }

    template<typename KeyT>
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        const gamax::size_type num_groups,
        gamax::mutable_column_view result_val,
        gamax::size_type* permutation_idx = nullptr) {

        using AggT = typename F::OutputType;
        
        auto num_items = key.size();
        auto result_key_ptr = gamax::make_column(key.type(), num_groups);
        auto d_num_groups = gamax::make_column(gamax::data_type{size_type_id}, 1);

        if(permutation_idx != nullptr) {
            auto val_itr = thrust::make_permutation_iterator(val.data<ValT>(), permutation_idx);
            CALL_CUB_FUNC(cub::DeviceReduce::ReduceByKey, key.data<KeyT>(), result_key_ptr->data<KeyT>(), val_itr, result_val.data<AggT>(), d_num_groups->data<gamax::size_type>(), F{}, num_items)
        } else {
            CALL_CUB_FUNC(cub::DeviceReduce::ReduceByKey, key.data<KeyT>(), result_key_ptr->data<KeyT>(), val.data<ValT>(), result_val.data<AggT>(), d_num_groups->data<gamax::size_type>(), F{}, num_items)
        }
    }
};

template<gamax::groupby::aggregation_id agg_kind>
struct dispatch_key_type_helper {
    template<typename ValT>
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        gamax::mutable_column_view result_key,
        gamax::mutable_column_view result_val,
        gamax::size_type* d_num_groups,
        gamax::size_type* permutation_idx = nullptr) {

        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, ValT>::F;
        
        gamax::type_dispatcher(
            key.type(), 
            reduce_by_key_aggregator_helper<F, ValT>{}, 
            key, 
            val, 
            result_key, 
            result_val, 
            d_num_groups,
            permutation_idx
        );
    }

    template<typename ValT>
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        const gamax::size_type num_groups,
        gamax::mutable_column_view result_val,
        gamax::size_type* permutation_idx = nullptr) {

        using F = typename agg_fn_mapper<agg_kind, ValT>::F;

        gamax::type_dispatcher(
            key.type(), 
            reduce_by_key_aggregator_helper<F, ValT>{}, 
            key, 
            val, 
            num_groups,
            result_val,
            permutation_idx
        );
    }
};

template<gamax::groupby::aggregation_id agg_kind>
struct dispatch_val_type_helper {
    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        gamax::mutable_column_view result_key,
        gamax::mutable_column_view result_val,
        gamax::size_type* d_num_groups,
        gamax::size_type* permutation_idx = nullptr) {
        
        gamax::type_dispatcher(
            val.type(), 
            dispatch_key_type_helper<agg_kind>{}, 
            key, 
            val, 
            result_key, 
            result_val, 
            d_num_groups,
            permutation_idx
        );
    }

    void operator() (
        gamax::column_view key, 
        gamax::column_view val, 
        const gamax::size_type num_groups,
        gamax::mutable_column_view result_val,
        gamax::size_type* permutation_idx = nullptr) {

        gamax::type_dispatcher(
            val.type(), 
            dispatch_key_type_helper<agg_kind>{}, 
            key, 
            val, 
            num_groups,
            result_val,
            permutation_idx
        );
    }
};

DISPATCH_AGGREGATE(dispatch_val_type_helper)

void aggregate(
    gamax::groupby::aggregation_id agg_kind, 
    gamax::column_view key, 
    gamax::column_view val, 
    gamax::mutable_column_view result_key,
    gamax::mutable_column_view result_val,
    gamax::size_type& num_groups,
    gamax::size_type* permutation_idx = nullptr) {

    auto d_num_groups = gamax::make_column(gamax::data_type{size_type_id}, 1);
    dispatch_aggregate(agg_kind, key, val, result_key, result_val, d_num_groups->data<gamax::size_type>(), permutation_idx);
    cudaMemcpy(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost);
}

void aggregate(
    gamax::groupby::aggregation_id agg_kind, 
    gamax::column_view key, 
    gamax::column_view val, 
    const gamax::size_type num_groups,
    gamax::mutable_column_view result_val,
    gamax::size_type* permutation_idx = nullptr) {

    dispatch_aggregate(agg_kind, key, val, num_groups, result_val, permutation_idx);
}

void check_solvable(gamax::table_view const& key, gamax::table_view const& val, std::vector<gamax::groupby::aggregation_request>& agg_requests) {
    if(key.num_columns() > 1) {
        throw std::runtime_error("Only single-column group keys are supported");
    }
    
    if(key.num_columns() == 0) {
        throw std::runtime_error("Aggregation only is not supported yet");
    }

    if(agg_requests.size() == 0) {
        throw std::runtime_error("No aggregation function is specified");
    }
}

void extract_groupby_hint(
    gamax::groupby::groupby_hint& hint, 
    gamax::size_type input_num_rows, 
    gamax::data_type dtype, 
    gamax::size_type& estimated_num_groups, 
    int& begin_bit, 
    int& end_bit) 
{
    begin_bit = 0;
    end_bit = 8 * gamax::size_of(dtype);
    estimated_num_groups = std::min(input_num_rows, hint.groups_upper_bound);
    
    if(hint.zero_based_ids) {
        end_bit = [&](gamax::size_type v) {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v++;
            return 8 * gamax::size_of(dtype) - __builtin_clz(v);
        } (estimated_num_groups);
    }
} 
} // unnamed namespace

std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream) 
{
    check_solvable(key, val, agg_requests);

    auto input_num_rows = key.num_rows();
    auto key_type = key.get_column(0).type();
    
    gamax::size_type estimated_num_groups;
    int begin_bit, end_bit;
    extract_groupby_hint(hint, input_num_rows, key_type, estimated_num_groups, begin_bit, end_bit);

    auto first_val_col_id = agg_requests[0].column_index;
    auto [sorted_key, sorted_val_0] = gamax::sort::sort_column_pair(key.get_column(0), val.get_column(first_val_col_id), begin_bit, end_bit);
    
    gamax::size_type num_groups = 0;
    std::vector<std::unique_ptr<gamax::column>> final_columns;

    std::unique_ptr<gamax::column> encoded_key = nullptr;

    bool grouped = false;
    for(auto& req: agg_requests) {
        auto i = req.column_index;
        auto agg_kind = req.agg_kind;
        std::unique_ptr<column> sorted_val = nullptr;
        auto val_type = val.get_column(i).type();
        
        if(!grouped) {
            sorted_val = std::move(sorted_val_0);
            auto unique_keys = gamax::make_column(key_type, estimated_num_groups);
            auto agg_val = gamax::make_column(val_type, estimated_num_groups);
            
            aggregate(agg_kind, sorted_key->view(), sorted_val->view(), unique_keys->mutable_view(), agg_val->mutable_view(), num_groups);

            unique_keys->set_size(num_groups);
            agg_val->set_size(num_groups);

            final_columns.push_back(std::move(unique_keys));
            final_columns.push_back(std::move(agg_val));

            grouped = true;
        }
        else {
            if(hint.optimize_sort) {
                if(encoded_key.get() == nullptr) {
                    encoded_key = gamax::compress::dictionary_encode(
                        key.get_column(0), 
                        // *(final_columns.front()), 
                        final_columns.front()->view(),
                        data_type{type_id::INT32}, 
                        stream);
                }
                auto end_bit = [&](gamax::size_type v) {
                    v--;
                    v |= v >> 1;
                    v |= v >> 2;
                    v |= v >> 4;
                    v |= v >> 8;
                    v |= v >> 16;
                    v++;
                    return 8 * gamax::size_of(encoded_key->type()) - __builtin_clz(v);
                } (num_groups);
                std::tie(std::ignore, sorted_val) = gamax::sort::sort_column_pair(encoded_key->view(), val.get_column(i), 0, end_bit);
            } else {
                std::tie(std::ignore, sorted_val) = gamax::sort::sort_column_pair(key.get_column(0), val.get_column(i), begin_bit, end_bit);
            }

            auto agg_val = gamax::make_column(val_type, num_groups);
            aggregate(agg_kind, sorted_key->view(), sorted_val->view(), num_groups, agg_val->mutable_view());
            final_columns.push_back(std::move(agg_val));
        }
    }

    cudaStreamSynchronize(stream);

    return std::make_unique<gamax::table>(std::move(final_columns));
}

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream) 
{
    // std::cout << "groupby_gfur" << std::endl;
    check_solvable(key, val, agg_requests);

    auto input_num_rows = key.num_rows();
    auto key_type = key.get_column(0).type();
    
    gamax::size_type estimated_num_groups;
    int begin_bit, end_bit;
    extract_groupby_hint(hint, input_num_rows, key_type, estimated_num_groups, begin_bit, end_bit);

    auto [sorted_key, idx_col] = gamax::sort::argsort(key.get_column(0), begin_bit, end_bit);
    
    gamax::size_type num_groups = 0;
    std::vector<std::unique_ptr<gamax::column>> final_columns;

    bool grouped = false;
    for(auto& req: agg_requests) {
        auto i = req.column_index;
        auto agg_kind = req.agg_kind;

        auto val_type = val.get_column(i).type();
        
        if(!grouped) {
            auto agg_val = gamax::make_column(val_type, estimated_num_groups);
            auto unique_keys = gamax::make_column(key_type, estimated_num_groups);
            aggregate(agg_kind, sorted_key->view(), val.get_column(i), unique_keys->mutable_view(), agg_val->mutable_view(), num_groups, idx_col->data<gamax::size_type>());
            unique_keys->set_size(num_groups);
            agg_val->set_size(num_groups);
            final_columns.push_back(std::move(unique_keys));
            final_columns.push_back(std::move(agg_val));

            grouped = true;
        } else {
            auto agg_val = gamax::make_column(val_type, num_groups);
            aggregate(agg_kind, sorted_key->view(), val.get_column(i), num_groups, agg_val->mutable_view(), idx_col->data<gamax::size_type>());
            final_columns.push_back(std::move(agg_val));
        }
    }

    cudaStreamSynchronize(0);

    return std::make_unique<gamax::table>(std::move(final_columns));
}

std::unique_ptr<gamax::table> groupby(
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    return groupby(key.view(), val.view(), agg_requests, hint, stream);
}

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    return groupby_gfur(key.view(), val.view(), agg_requests, hint, stream);
}

} // namespace gamax::groupby::sort
