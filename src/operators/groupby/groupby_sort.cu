#include <memory>
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
} // unnamed namespace

// TODO
// 1. support the aggregation function in a better way
std::unique_ptr<gamax::table> groupby(
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

    auto [sorted_key, sorted_val_0] = gamax::sort::sort_column_pair(key.get_column(0), val.get_column(0));
    
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

        std::unique_ptr<column> sorted_val = nullptr;
        if(i == 0) {
            sorted_val = std::move(sorted_val_0);
        }
        else {
            auto sorted_pair = gamax::sort::sort_column_pair(key.get_column(i), val.get_column(i));
            sorted_val = std::move(sorted_pair.second);
            gamax::free_column(std::move(sorted_pair.first));
        }
        if(agg_kind == gamax::groupby::aggregation_id::SUM) {
            if(val.get_column(i).type().id() == gamax::type_id::INT32) {
                using agg_fn_t = gamax::groupby::Sum<int32_t>;
                using agg_output_t = agg_fn_t::OutputType;
                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, num_groups, true);
                aggregate<agg_fn_t>(sorted_val->data<agg_fn_t::InputType>(), input_num_rows, num_units, units_start, units_end, units_group, num_groups, agg_val->data<agg_fn_t::OutputType>());
                final_columns.push_back(std::move(agg_val));
            }
            else if(val.get_column(i).type().id() == gamax::type_id::INT64) {
                using agg_fn_t = gamax::groupby::Sum<int64_t>;
                using agg_output_t = agg_fn_t::OutputType;
                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, num_groups);
                aggregate<agg_fn_t>(sorted_val->data<agg_fn_t::InputType>(), input_num_rows, num_units, units_start, units_end, units_group, num_groups, agg_val->data<agg_fn_t::OutputType>());
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
                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, num_groups);
                aggregate<agg_fn_t>(reinterpret_cast<agg_fn_t::InputType*>(sorted_val->data()), input_num_rows, num_units, units_start, units_end, units_group, num_groups, reinterpret_cast<agg_fn_t::OutputType*>(agg_val->data()));
                final_columns.push_back(std::move(agg_val));
            }
            else if(val.get_column(i).type().id() == gamax::type_id::INT64) {
                using agg_fn_t = gamax::groupby::Max<int64_t>;
                using agg_output_t = agg_fn_t::OutputType;
                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, num_groups);
                aggregate<agg_fn_t>(reinterpret_cast<agg_fn_t::InputType*>(sorted_val->data()), input_num_rows, num_units, units_start, units_end, units_group, num_groups, reinterpret_cast<agg_fn_t::OutputType*>(agg_val->data()));
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

std::unique_ptr<gamax::table> groupby_gfur(
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
} // namespace gamax::groupby::sort
