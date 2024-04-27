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
#include "operators/groupby/load_balancer.cuh"
#include "operators/groupby/agg_func.cuh"
#include "operators/sort/sort.hpp"
#include "types/data_type.hpp"

namespace gamax::groupby::sort {

namespace {

template<typename F, typename ValT>
struct reduce_by_key_aggregator_helper {
    template<typename KeyT>
    void operator() (
        gamax::column& key, 
        gamax::column& val, 
        gamax::column& result_key,
        gamax::column& result_val,
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
        gamax::column& key, 
        gamax::column& val, 
        const gamax::size_type num_groups,
        gamax::column& result_val,
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
        gamax::column& key, 
        gamax::column& val, 
        gamax::column& result_key,
        gamax::column& result_val,
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
        gamax::column& key, 
        gamax::column& val, 
        const gamax::size_type num_groups,
        gamax::column& result_val,
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
        gamax::column& key, 
        gamax::column& val, 
        gamax::column& result_key,
        gamax::column& result_val,
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
        gamax::column& key, 
        gamax::column& val, 
        const gamax::size_type num_groups,
        gamax::column& result_val,
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

template<typename... Ts>
void dispatch_aggregate(gamax::groupby::aggregation_id agg_kind, Ts&&... args) {
    switch(agg_kind) {
        case gamax::groupby::aggregation_id::SUM:
            dispatch_val_type_helper<gamax::groupby::aggregation_id::SUM>{}(std::forward<Ts>(args)...);
            break;
        case gamax::groupby::aggregation_id::MAX:
            dispatch_val_type_helper<gamax::groupby::aggregation_id::MAX>{}(std::forward<Ts>(args)...);
            break;
        default:
            throw std::runtime_error("Unsupported aggregation function");
    }
}

void aggregate(
    gamax::groupby::aggregation_id agg_kind, 
    gamax::column& key, 
    gamax::column& val, 
    gamax::column& result_key,
    gamax::column& result_val,
    gamax::size_type& num_groups,
    gamax::size_type* permutation_idx = nullptr) {

    auto d_num_groups = gamax::make_column(gamax::data_type{size_type_id}, 1);
    dispatch_aggregate(agg_kind, key, val, result_key, result_val, d_num_groups->data<gamax::size_type>(), permutation_idx);
    cudaMemcpy(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost);
}

void aggregate(
    gamax::groupby::aggregation_id agg_kind, 
    gamax::column& key, 
    gamax::column& val, 
    const gamax::size_type num_groups,
    gamax::column& result_val,
    gamax::size_type* permutation_idx = nullptr) {

    dispatch_aggregate(agg_kind, key, val, num_groups, result_val, permutation_idx);
}

void check_solvable(gamax::table& key, gamax::table& val, std::vector<gamax::groupby::aggregation_request>& agg_requests) {
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
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint) 
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
            
            aggregate(agg_kind, *sorted_key, *sorted_val, *unique_keys, *agg_val, num_groups);

            unique_keys->set_size(num_groups);
            agg_val->set_size(num_groups);

            final_columns.push_back(std::move(unique_keys));
            final_columns.push_back(std::move(agg_val));

            grouped = true;
        }
        else {
            auto sorted_pair = gamax::sort::sort_column_pair(key.get_column(i), val.get_column(i), begin_bit, end_bit);
            sorted_val = std::move(sorted_pair.second);
            gamax::free_column(std::move(sorted_pair.first));

            auto agg_val = gamax::make_column(val_type, num_groups);
            aggregate(agg_kind, *sorted_key, *sorted_val, num_groups, *agg_val);
            final_columns.push_back(std::move(agg_val));
        }
    }

    cudaStreamSynchronize(0);

    return std::make_unique<gamax::table>(std::move(final_columns));
}

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint) 
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
            aggregate(agg_kind, *sorted_key, val.get_column(i), *unique_keys, *agg_val, num_groups, idx_col->data<gamax::size_type>());
            unique_keys->set_size(num_groups);
            agg_val->set_size(num_groups);
            final_columns.push_back(std::move(unique_keys));
            final_columns.push_back(std::move(agg_val));

            grouped = true;
        } else {
            auto agg_val = gamax::make_column(val_type, num_groups);
            aggregate(agg_kind, *sorted_key, val.get_column(i), num_groups, *agg_val, idx_col->data<gamax::size_type>());
            final_columns.push_back(std::move(agg_val));
        }
    }

    cudaStreamSynchronize(0);

    return std::make_unique<gamax::table>(std::move(final_columns));
}

/*
namespace {
    template<typename T, 
        typename NumItemsT = gamax::size_type, 
        typename IdxT = NumItemsT>
void group_keys(T* sorted_key, 
                T* unique_keys_out, 
                NumItemsT num_items,
                NumItemsT* num_groups, 
                IdxT* membership_ps) {
    IdxT* membership_cnt = nullptr;
    allocate_mem(&membership_cnt, true, num_items*sizeof(IdxT));
    NumItemsT* num_runs = nullptr;
    allocate_mem(&num_runs, true, sizeof(NumItemsT));
    CALL_CUB_FUNC(cub::DeviceRunLengthEncode::Encode, sorted_key, unique_keys_out, membership_cnt, num_runs, num_items)

    cudaMemcpy(num_groups, num_runs, sizeof(NumItemsT), cudaMemcpyDeviceToHost);
    CALL_CUB_FUNC(cub::DeviceScan::InclusiveSum, membership_cnt, membership_ps, *num_groups)

    release_mem(membership_cnt);
    release_mem(num_runs);
}

struct group_keys_helper {
    template<typename T, 
             typename NumItemsT = gamax::size_type, 
             typename IdxT = NumItemsT>
    void operator()(
        gamax::column const& sorted_key, 
        gamax::column& unique_keys_out, 
        NumItemsT num_items,
        NumItemsT* num_groups, 
        gamax::column& membership_ps) {
        
        group_keys(sorted_key.data<T>(), unique_keys_out.data<T>(), num_items, num_groups, membership_ps.data<IdxT>());
    }
};

void group_keys(
    gamax::column const& sorted_key,
    gamax::column& unique_keys_out,
    gamax::size_type num_items,
    gamax::size_type* num_groups,
    gamax::column& membership_ps) {
        
    gamax::type_dispatcher(
        sorted_key.type(), 
        group_keys_helper{}, 
        sorted_key, 
        unique_keys_out, 
        num_items, 
        num_groups, 
        membership_ps
    );
}

template<typename F, int NT, typename ValIt, typename NumItemsT, typename IdxT, typename ResultT>
__global__ void aggregate_kernel_block_monoid(ValIt values, 
                NumItemsT num_items, 
                NumItemsT num_units,
                IdxT* units_start, 
                IdxT* units_end, 
                IdxT* units_group, 
                const NumItemsT num_groups, 
                ResultT* final_results) {
    static_assert(F::is_monoid);
    
    F f;
    ResultT thread_local_agg = F::identity;
    
    #pragma unroll
    for(int i = blockIdx.x; i < num_units; i += gridDim.x) {
        IdxT group_id = units_group[i];
        
        #pragma unroll
        for(int t = units_start[i]+threadIdx.x; t < units_end[i]; t += blockDim.x) {
            thread_local_agg = f.thread_local_update(thread_local_agg, values[t]);
        }
        __syncthreads();
        
        // block reduction
        typedef cub::BlockReduce<decltype(thread_local_agg), NT> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        ResultT block_agg = BlockReduce(temp_storage).Reduce(thread_local_agg, f);
        if(threadIdx.x == 0) { // only thread-0 has the correct aggregate
            f.global_update(group_id, final_results, block_agg);
        }

        thread_local_agg = F::identity;
    }
}

template<typename F, typename ValIt = typename F::InputType*>
void aggregate(ValIt values, 
                gamax::size_type num_items, 
                gamax::size_type num_units,
                gamax::size_type* units_start, 
                gamax::size_type* units_end, 
                gamax::size_type* units_group, 
                const gamax::size_type num_groups,
                typename F::OutputType* final_results) {
    // each package: (start, end, group_id)
    // assign one thread block per work package
    constexpr int NT = 256;
    aggregate_kernel_block_monoid<F, NT><<<num_units, NT>>>(values, num_items, num_units, units_start, units_end, units_group, num_groups, final_results);
}

}
std::unique_ptr<gamax::table> groupby_gfur_old(
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint) 
{
    if(key.num_columns() > 1) {
        throw std::runtime_error("Only single-column group keys are supported");
    }
    
    if(key.num_columns() == 0) {
        throw std::runtime_error("Aggregation only is not supported yet");
    }

    auto input_num_rows = key.num_rows();
    auto estimated_num_groups = std::min(input_num_rows, hint.groups_upper_bound);

    auto [sorted_key, idx_col] = gamax::sort::argsort(key.get_column(0));
    
    gamax::size_type num_groups = 0;
    auto unique_keys = gamax::make_column(key.get_column(0).type(), estimated_num_groups);
    auto membership_ps_ptr = gamax::make_column(gamax::data_type{type_to_id<gamax::size_type>()}, estimated_num_groups);
    group_keys(*sorted_key, *unique_keys, input_num_rows, &num_groups, *membership_ps_ptr);
    unique_keys->set_size(num_groups);
    
    gamax::free_column(std::move(sorted_key));

    std::vector<std::unique_ptr<gamax::column>> final_columns;
    final_columns.push_back(std::move(unique_keys));

    constexpr gamax::size_type threshold = 4096;
    int num_units = 0;
    int max_work_units = num_groups + (input_num_rows+threshold-1)/threshold;

    auto units_start_ptr = gamax::make_idx_column(max_work_units);
    auto units_end_ptr = gamax::make_idx_column(max_work_units);
    auto units_group_ptr = gamax::make_idx_column(max_work_units);

    auto units_start = units_start_ptr->data<gamax::size_type>();
    auto units_end = units_end_ptr->data<gamax::size_type>();
    auto units_group = units_group_ptr->data<gamax::size_type>();

    generate_workunits(reinterpret_cast<gamax::size_type*>(membership_ps_ptr->data()), num_groups, threshold, &num_units, units_start, units_end, units_group);

    gamax::free_column(std::move(membership_ps_ptr));

    for(auto& req: agg_requests) {
        auto i = req.column_index;
        auto agg_kind = req.agg_kind;

        auto& val_col = val.get_column(i);
        
        if(agg_kind == gamax::groupby::aggregation_id::SUM) {
            if(val.get_column(i).type().id() == gamax::type_id::INT32) {
                using agg_fn_t = gamax::groupby::Sum<int32_t>;
                using agg_output_t = agg_fn_t::OutputType;
                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, num_groups, true);

                auto val_data = val_col.data<agg_fn_t::InputType>();
                auto val_itr = thrust::make_permutation_iterator(val_data, idx_col->data<gamax::size_type>());

                aggregate<agg_fn_t>(val_itr, input_num_rows, num_units, units_start, units_end, units_group, num_groups, agg_val->data<agg_fn_t::OutputType>());
                final_columns.push_back(std::move(agg_val));
            }
            else if(val.get_column(i).type().id() == gamax::type_id::INT64) {
                using agg_fn_t = gamax::groupby::Sum<int64_t>;
                using agg_output_t = agg_fn_t::OutputType;
                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, num_groups, true);

                auto val_data = val_col.data<agg_fn_t::InputType>();
                auto val_itr = thrust::make_permutation_iterator(val_data, idx_col->data<gamax::size_type>());

                aggregate<agg_fn_t>(val_itr, input_num_rows, num_units, units_start, units_end, units_group, num_groups, agg_val->data<agg_fn_t::OutputType>());
                final_columns.push_back(std::move(agg_val));
            }
            else {
                throw std::runtime_error("Unsupported data type for SUM aggregation");
            }
        }
        else if(agg_kind == gamax::groupby::aggregation_id::MAX) {
            if(val.get_column(i).type().id() == gamax::type_id::INT32) {
                using agg_fn_t = gamax::groupby::Max<int32_t>;
                using agg_output_t = agg_fn_t::OutputType;
                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, num_groups, true);

                auto val_data = val_col.data<agg_fn_t::InputType>();
                auto val_itr = thrust::make_permutation_iterator(val_data, idx_col->data<gamax::size_type>());

                aggregate<agg_fn_t>(val_itr, input_num_rows, num_units, units_start, units_end, units_group, num_groups, agg_val->data<agg_fn_t::OutputType>());
                final_columns.push_back(std::move(agg_val));
            }
            else if(val.get_column(i).type().id() == gamax::type_id::INT64) {
                using agg_fn_t = gamax::groupby::Max<int64_t>;
                using agg_output_t = agg_fn_t::OutputType;
                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, num_groups, true);

                auto val_data = val_col.data<agg_fn_t::InputType>();
                auto val_itr = thrust::make_permutation_iterator(val_data, idx_col->data<gamax::size_type>());

                aggregate<agg_fn_t>(val_itr, input_num_rows, num_units, units_start, units_end, units_group, num_groups, agg_val->data<agg_fn_t::OutputType>());
                final_columns.push_back(std::move(agg_val));
            }
            else {
                throw std::runtime_error("Unsupported data type for SUM aggregation");
            }
        }
        else {
            throw std::runtime_error("Unsupported aggregation function");
        }
    }

    cudaStreamSynchronize(0);

    return std::make_unique<gamax::table>(std::move(final_columns));
}
*/
} // namespace gamax::groupby::sort
