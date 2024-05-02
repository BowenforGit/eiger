#include "operators/groupby/groupby_partition.hpp"

#include <memory>
#include <cuda.h>
#include <type_traits>
#include <thrust/iterator/permutation_iterator.h>
#include <cuco/static_map.cuh>

#include "utils/utils.cuh"
#include "containers/agg_hashmap.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/table_device_view.hpp"
#include "operators/partition/partition.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/load_balancer.cuh"
#include "operators/groupby/agg_func.cuh"
#include "types/data_type.hpp"

namespace gamax::groupby::partition {
namespace {

template<gamax::groupby::aggregation_id agg_kind>
struct dispatch_val_type_helper {
    template<typename T>
    __device__ __forceinline__ void operator()(
        gamax::column& src,
        gamax::size_type src_idx,
        gamax::column& dst,
        gamax::size_type dst_idx)
    {
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;
        F{}(src, src_idx, dst, dst_idx);
    }

    __device__ __forceinline__ void operator()(
        gamax::column& src,
        gamax::size_type src_idx,
        gamax::column& dst,
        gamax::size_type dst_idx)
    {

        type_dispatcher(
            src.type(),
            dispatch_val_type_helper<agg_kind>{},
            src, 
            src_idx, 
            dst, 
            dst_idx);
    }
};

template<typename... Ts>
__device__ __forceinline__ void dispatch_type_aggregate(gamax::groupby::aggregation_id agg_kind, Ts&&... args)
{
    switch(agg_kind) {
        case gamax::groupby::aggregation_id::SUM:
            dispatch_val_type_helper<gamax::groupby::aggregation_id::SUM>{}(std::forward<Ts>(args)...);
            break;
        case gamax::groupby::aggregation_id::MAX:
            dispatch_val_type_helper<gamax::groupby::aggregation_id::MAX>{}(std::forward<Ts>(args)...);
            break;
        default:
            GAMAX_UNREACHABLE("Unsupported aggregation function");
    }
}

__device__ __forceinline__ void aggregate(
    gamax::groupby::aggregation_id agg_kind,
    gamax::column& src,
    gamax::size_type src_idx,
    gamax::column& dst,
    gamax::size_type dst_idx)
{
    dispatch_type_aggregate(
        agg_kind,
        src, 
        src_idx, 
        dst, 
        dst_idx);
}

template<typename Key, 
         std::size_t NumWindows, 
         typename IdxIt,
         int cg_size = 1,
         int window_size = 1>
__global__ void partition_groupby_aggregate_kernel(Key* keys,
                                            table_device_view values, 
                                            IdxIt val_idx,
                                            const gamax::size_type num_units,
                                            gamax::size_type* units_start, 
                                            gamax::size_type* units_end, 
                                            gamax::size_type* num_groups,
                                            gamax::size_type* group_id,
                                            gamax::groupby::aggregation_request* aggs,
                                            const gamax::size_type num_aggs,
                                            table_device_view res) 
{
    constexpr int max_partition_size = 4096;
    constexpr gamax::size_type groupid_write_mask = 0x80000000;
    constexpr gamax::size_type groupid_read_mask = 0x7FFFFFFF;

    using Value     = gamax::size_type; // tuple id
    using slot_type = cuco::pair<Key, Value>;
    using extent_type      = cuco::extent<std::size_t, NumWindows>;
    using storage_ref_type = cuco::aow_storage_ref<slot_type, window_size, extent_type>;

    extern __shared__ int16_t temp[];
    auto map = reinterpret_cast<cuco::window<slot_type, window_size>*>(temp);
    union {
        gamax::size_type num_groups_local;
        gamax::size_type block_offset;
    } *u = reinterpret_cast<decltype(u)>(map+NumWindows);
    
    auto group_loc = group_id;

    auto key_arr = keys;

    auto const block = cooperative_groups::this_thread_block();
    
    for(int i = blockIdx.x; i < num_units; i += gridDim.x) {
        if(block.thread_rank() == 0) u->num_groups_local = 0;

        auto raw_ref =
            cuco::static_map_ref{cuco::empty_key<Key>{-1}, // TODO: reconsider the sentinel value
                                cuco::empty_value<Value>{-1},
                                thrust::equal_to<Key>{},
                                cuco::linear_probing<cg_size, cuco::default_hash_function<Key>>{},
                                cuco::thread_scope_block,
                                storage_ref_type{extent_type{}, map}};
        raw_ref.initialize(block);

        auto insert_and_find_ref = std::move(raw_ref).with(cuco::op::insert_and_find);

        #pragma unroll
        for(int t = block.thread_rank(); t < units_end[i] - units_start[i]; t += block.size()) {
            auto [slot, is_new_key] = insert_and_find_ref.insert_and_find(cuco::pair{key_arr[t+units_start[i]], t+units_start[i]});
            if(is_new_key) {
                auto t_group_id = atomicAdd(&(u->num_groups_local), 1); // block-scope group id
                group_id[t+units_start[i]] = (groupid_write_mask | t_group_id); // flag that his is a group id not a location
                // printf("t_group_id: %d, %d\n", t_group_id, group_id[t+units_start[i]]);
            } else {
                group_loc[t+units_start[i]] = slot->second;
            }
        }
        block.sync();

        if(block.thread_rank() == 0) {
            // printf("num_groups_local: %d\n", u->num_groups_local);
            u->block_offset = atomicAdd(num_groups, u->num_groups_local);
        }
        block.sync();

        for(int t = block.thread_rank(); t < units_end[i] - units_start[i]; t += block.size()) {
            gamax::size_type g_group_id {-1}; // global group id
            auto gl = group_loc[t+units_start[i]];
            if((gl & groupid_write_mask) == groupid_write_mask) { // the first inserter
                g_group_id = u->block_offset + (gl & groupid_read_mask);
                res[0].data<Key>()[g_group_id] = key_arr[t+units_start[i]];
            } else {
                // printf("gl: %d, units_end[i]: %d\n", gl, units_end[i]);
                // if((group_id[gl] & groupid_read_mask) != 0) printf("%d %d\n", group_id[gl], (group_id[gl] & groupid_read_mask));
                g_group_id = u->block_offset + (group_id[gl] & groupid_read_mask);
            }

            #pragma unroll
            for(gamax::size_type k = 0; k < num_aggs; k++) {
                aggregate(
                    aggs[k].agg_kind,
                    values.get_column(aggs[k].column_index), val_idx[t+units_start[i]],
                    res.get_column(k+1), g_group_id);
            }
        }
        block.sync();
    }
}

template<int window_size, int numWindows>
struct partition_groupby_impl {
    template<typename Key>
    std::unique_ptr<gamax::table> operator()(
            gamax::table& key,
            gamax::table& val,
            std::vector<gamax::groupby::aggregation_request>& agg_requests,
            gamax::groupby::groupby_hint& hint,
            const int log_num_partitions,
            cudaStream_t stream = cudaStreamDefault)
        {
            auto input_num_rows = key.num_rows();
            gamax::size_type estimated_num_groups = std::min(input_num_rows, hint.groups_upper_bound);

            auto d_agg_req_buf = std::make_unique<gamax::device_buffer>(sizeof(gamax::groupby::aggregation_request) * agg_requests.size(), false);
            cudaMemcpy(d_agg_req_buf->data(), agg_requests.data(), d_agg_req_buf->size(), cudaMemcpyHostToDevice);
            
            int num_partitions = 1 << log_num_partitions;

            std::vector<std::unique_ptr<gamax::column>> partitioned_val_cols;
            
            std::unique_ptr<gamax::column> partitioned_key {nullptr};
            std::unique_ptr<gamax::column> pval_temp {nullptr};
            for(int i = 0; i < val.num_columns(); i++) {
                // when assigning to the partitioned key, the previous pointed resource should be released right?
                std::tie(partitioned_key, pval_temp) = gamax::partition::partition_column_pair(key.get_column(0), val.get_column(i), 0, log_num_partitions);
                partitioned_val_cols.push_back(std::move(pval_temp));
                // val.free_column(i);
            }
            auto partitioned_val = std::make_unique<gamax::table>(std::move(partitioned_val_cols));
            // print_gpu_arr(partitioned_key->data<Key>(), 16);

            constexpr gamax::size_type threshold = std::numeric_limits<gamax::size_type>::max() / 2; // TODO: be careful of overflow
            int max_work_units = num_partitions + (input_num_rows+threshold-1)/threshold;

            auto units_start_ptr = gamax::make_idx_column(max_work_units);
            auto units_end_ptr = gamax::make_idx_column(max_work_units);
            auto units_group_ptr = gamax::make_idx_column(max_work_units);

            auto units_start = units_start_ptr->data<gamax::size_type>();
            auto units_end = units_end_ptr->data<gamax::size_type>();
            auto units_group = units_group_ptr->data<gamax::size_type>();

            int num_units = 0;
            
            {
                auto partition_ps = gamax::partition::compute_partition_offsets(*partitioned_key, 0, log_num_partitions);
                gamax::groupby::generate_workunits(partition_ps->data<gamax::size_type>(), num_partitions, threshold, &num_units, units_start, units_end, units_group);
            }

            std::vector<std::unique_ptr<gamax::column>> final_columns;
            final_columns.push_back(
                gamax::make_column(
                    key.get_column(0).type(), 
                    estimated_num_groups, 
                    false));
            for(const auto& agg : agg_requests) {
                final_columns.push_back(
                    gamax::make_column(
                        gamax::groupby::agg_output_data_type(agg.agg_kind, val.get_column(agg.column_index).type()), 
                        estimated_num_groups, 
                        true)); // TODO: fix this initial aggregation value
            }
            auto res = std::make_unique<gamax::table>(std::move(final_columns));

            auto d_num_groups = gamax::make_idx_column(1, true);
            auto val_view = table_device_view::create(*partitioned_val);
            auto res_view = table_device_view::create(*res);
            auto group_id = gamax::make_idx_column(input_num_rows, false);

            std::size_t sm_bytes = numWindows * sizeof(cuco::window<cuco::pair<Key, gamax::size_type>, window_size>) + sizeof(gamax::size_type);
            std::cout << "sm_bytes: " << sm_bytes << std::endl; 
            auto itr = thrust::make_counting_iterator<gamax::size_type>(0);
            auto agg_fn = partition_groupby_aggregate_kernel<Key, numWindows, decltype(itr)>;

            if(sm_bytes > 48*1024) {
                CHECK_CUDA_ERROR(cudaFuncSetAttribute(agg_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes));
            }
            
            // the grid size and block size given by cudaOccupancyMaxPotentialBlockSize 
            // do not work very well for this kernel
            agg_fn<<<82*3, 512, sm_bytes, stream>>>(
                partitioned_key->data<Key>(),
                *val_view,
                itr,
                num_units,
                units_start,
                units_end,
                d_num_groups->data<gamax::size_type>(),
                group_id->data<gamax::size_type>(),
                static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()),
                agg_requests.size(),
                *res_view);
            
            gamax::size_type num_groups;
            cudaMemcpyAsync(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
            res->set_rows(num_groups);
            
            cudaStreamSynchronize(stream);

            return res;
        }
    };

template<typename F, 
            typename KeyT, 
            typename ValIt, 
            typename NumItemsT, 
            typename IdxT, 
            typename ResultT, 
            int VT=4>
__global__ void aggregate_kernel_block_monoid(KeyT* keys,
                                            ValIt values, 
                                            NumItemsT num_units,
                                            IdxT* units_start, 
                                            IdxT* units_end, 
                                            IdxT* units_group, 
                                            const int log_parts,
                                            NumItemsT* num_groups,
                                            KeyT* grouped_keys,
                                            ResultT* final_results) {
    static_assert(F::is_monoid && std::is_same_v<ResultT, typename F::OutputType>);
    
    extern __shared__ int16_t temp[];

    int max_partition_size = 4096;
    int hash_table_capacity = max_partition_size+512;
    auto hash_fn = [&](const KeyT& k) -> int {
        return (k >> log_parts) & (hash_table_capacity - 1);
    };
    using ValT = std::conditional_t<std::is_pointer_v<ValIt>, 
                    std::remove_reference_t<std::remove_pointer_t<ValIt>>, 
                    std::remove_reference_t<decltype(values[0])>>;
    using HashMapT = gamax::AggHashMap<KeyT, ValT, decltype(hash_fn), F>;
    HashMapT hash_map((void*)temp, hash_table_capacity, hash_fn); // todo: fix 4096

    #pragma unroll
    for(int i = blockIdx.x; i < num_units; i += gridDim.x) {
        // re-init for every group
        hash_map.Init();

        IdxT group_id = units_group[i];
        auto unit_len = units_end[i] - units_start[i];
        
        // aggregate into the block local hash table in the shared memory
        for(int t = units_start[i]; t < units_end[i]; t += VT*blockDim.x) {
            // printf("t: %d\n", t);
            KeyT tlkeys[VT];
            ValT tlvals[VT];
            
            #pragma unroll
            for(int k = 0; k < VT; k++) {
                tlkeys[k] = keys[t + VT*threadIdx.x + k];
                tlvals[k] = values[t + VT*threadIdx.x + k];
            }
            
            int tl_n_items = unit_len - (t-units_start[i]) - VT * threadIdx.x;              

            #pragma unroll
            for (int k = 0; k < VT; k++) {
                if (k < tl_n_items) {
                    auto e = hash_map.Insert(tlkeys[k], tlvals[k]);
                    // if(e == 1) printf("insert failed\n");
                }
            }
        }
        __syncthreads();

        F f;
        // aggregate the block local hash table into the global hash table
        for(int j = threadIdx.x; j < hash_table_capacity; j += blockDim.x) {
            if(!hash_map.is_empty(j)) {
                auto p = hash_map[j];
                // auto key = p.first;
                // auto local_agg = p.second;
                auto g = atomicAdd(num_groups, 1);
                grouped_keys[g] = p.first;
                final_results[g] = p.second;
            }
        }
        __syncthreads();
    }
}

template<typename F, 
            typename KeyT, 
            typename ValIt, 
            typename NumItemsT, 
            typename IdxT, 
            typename ResultT>
void aggregate(KeyT* keys,
                ValIt values, 
                NumItemsT num_units,
                IdxT* units_start, 
                IdxT* units_end, 
                IdxT* units_group, 
                const int log_parts,
                NumItemsT* num_groups,
                KeyT* grouped_keys,
                ResultT* final_results) {
    constexpr int NT = 256;

    NumItemsT* d_num_groups = nullptr;
    allocate_mem(&d_num_groups, true, sizeof(NumItemsT));
    size_t sm_bytes = (4096*4*2*2); // TODO: Fix this
    auto agg_fn = aggregate_kernel_block_monoid<F, KeyT, ValIt, NumItemsT, IdxT, ResultT, 1>;
    cudaFuncSetAttribute(agg_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes);
    agg_fn<<<num_units, NT, sm_bytes>>>(keys, values, 
                                                        num_units, units_start, 
                                                        units_end, units_group, 
                                                        log_parts, d_num_groups,
                                                        grouped_keys, final_results);
    cudaMemcpy(num_groups, d_num_groups, sizeof(NumItemsT), cudaMemcpyDeviceToHost);
    release_mem(d_num_groups);
}
} // unnamed namespace

std::unique_ptr<gamax::table> groupby_old(
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

    int log_num_partitions = 16;
    int num_partitions = 1 << log_num_partitions;

    auto [partitioned_key_0, partitioned_val_0] = gamax::partition::partition_column_pair(key.get_column(0), val.get_column(0), 0, log_num_partitions);

    constexpr gamax::size_type threshold = std::numeric_limits<gamax::size_type>::max() / 2; // TODO: be careful of overflow
    int max_work_units = num_partitions + (input_num_rows+threshold-1)/threshold;

    auto units_start_ptr = gamax::make_idx_column(max_work_units);
    auto units_end_ptr = gamax::make_idx_column(max_work_units);
    auto units_group_ptr = gamax::make_idx_column(max_work_units);

    auto units_start = units_start_ptr->data<gamax::size_type>();
    auto units_end = units_end_ptr->data<gamax::size_type>();
    auto units_group = units_group_ptr->data<gamax::size_type>();

    int num_units = 0;
    
    {
        auto partition_ps = gamax::partition::compute_partition_offsets(*partitioned_key_0, 0, log_num_partitions);
        gamax::groupby::generate_workunits(partition_ps->data<gamax::size_type>(), num_partitions, threshold, &num_units, units_start, units_end, units_group);
    }

    std::vector<std::unique_ptr<gamax::column>> final_columns;

    for(auto& req: agg_requests) {
        auto i = req.column_index;
        auto agg_kind = req.agg_kind;

        std::unique_ptr<column> partitioned_val = nullptr;
        std::unique_ptr<column> partitioned_key = nullptr;
        if(i == 0) {
            partitioned_key = std::move(partitioned_key_0);
            partitioned_val = std::move(partitioned_val_0);
        }
        else {
            auto partitioned_pair = gamax::partition::partition_column_pair(key.get_column(i), val.get_column(i), 0, log_num_partitions);
            partitioned_key = std::move(partitioned_pair.first);
            partitioned_val = std::move(partitioned_pair.second);
        }

        auto key_type = key.get_column(0).type();

        gamax::size_type num_groups = 0;
        auto grouped_key = std::make_unique<gamax::column>(key.get_column(0).type(), estimated_num_groups, true);
        std::unique_ptr<gamax::column> res_val = nullptr;
        
        if(agg_kind == gamax::groupby::aggregation_id::SUM) {
            if(val.get_column(i).type().id() == gamax::type_id::INT32) {
                using agg_fn_t = gamax::groupby::Sum<int32_t>;
                using agg_output_t = agg_fn_t::OutputType;

                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, estimated_num_groups, true);

                if(key_type == gamax::data_type{gamax::type_id::INT32}) {
                    aggregate<agg_fn_t>(partitioned_key->data<int32_t>(), partitioned_val->data<agg_fn_t::InputType>(), 
                                    num_units, units_start, units_end, units_group, log_num_partitions, 
                                    &num_groups,
                                    grouped_key->data<int32_t>(), agg_val->data<agg_fn_t::OutputType>());
                }
                else if(key_type == gamax::data_type{gamax::type_id::INT64}) {
                    aggregate<agg_fn_t>(partitioned_key->data<int64_t>(), partitioned_val->data<agg_fn_t::InputType>(), 
                                    num_units, units_start, units_end, units_group, log_num_partitions, 
                                    &num_groups,
                                    grouped_key->data<int64_t>(), agg_val->data<agg_fn_t::OutputType>());
                }
                else {
                    throw std::runtime_error("Unsupported key type for SUM aggregation");
                }
                
                res_val = std::move(agg_val);
            }
            else {
                throw std::runtime_error("Unsupported value type for SUM aggregation");
            }
        }
        else if(agg_kind == gamax::groupby::aggregation_id::MAX) {
            if(val.get_column(i).type().id() == gamax::type_id::INT32) {
                using agg_fn_t = gamax::groupby::Max<int32_t>;
                using agg_output_t = agg_fn_t::OutputType;

                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, estimated_num_groups, true);

                if(key_type == gamax::data_type{gamax::type_id::INT32}) {
                    aggregate<agg_fn_t>(partitioned_key->data<int32_t>(), partitioned_val->data<agg_fn_t::InputType>(), 
                                    num_units, units_start, units_end, units_group, log_num_partitions, 
                                    &num_groups,
                                    grouped_key->data<int32_t>(), agg_val->data<agg_fn_t::OutputType>());
                }
                else if(key_type == gamax::data_type{gamax::type_id::INT64}) {
                    aggregate<agg_fn_t>(partitioned_key->data<int64_t>(), partitioned_val->data<agg_fn_t::InputType>(), 
                                    num_units, units_start, units_end, units_group, log_num_partitions, 
                                    &num_groups,
                                    grouped_key->data<int64_t>(), agg_val->data<agg_fn_t::OutputType>());
                }
                else {
                    throw std::runtime_error("Unsupported key type for MAX aggregation");
                }
                
                res_val = std::move(agg_val);
            }
            else {
                throw std::runtime_error("Unsupported value type for MAX aggregation");
            }
        }
        else {
            throw std::runtime_error("Unsupported aggregation function");
        }

        if(final_columns.size() == 0) {
            grouped_key->set_size(num_groups);
            final_columns.push_back(std::move(grouped_key));
        }
        res_val->set_size(num_groups);
        final_columns.push_back(std::move(res_val));
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
    constexpr auto log_num_partitions = 16;
    constexpr auto num_partitions = 1 << log_num_partitions;
    constexpr auto cg_size = 1;
    constexpr auto window_size = 1;
    constexpr auto numWindows = 4096;
    
    auto constexpr num_windows = cuco::make_window_extent<cg_size, window_size>(cuco::extent<gamax::size_type, numWindows>{});
    return type_dispatcher(
        key.get_column(0).type(),
        partition_groupby_impl<window_size, num_windows.value()>{},
        key, val, agg_requests, hint, log_num_partitions, stream);
}

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table& key, 
    gamax::table& val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    if(key.num_columns() > 1) {
        throw std::runtime_error("Only single-column group keys are supported");
    }
    
    if(key.num_columns() == 0) {
        throw std::runtime_error("Aggregation only is not supported yet");
    }

    auto input_num_rows = key.num_rows();
    auto estimated_num_groups = std::min(input_num_rows, hint.groups_upper_bound);

    int log_num_partitions = 16;
    int num_partitions = 1 << log_num_partitions;

    auto [partitioned_key, idx_col] = gamax::partition::arg_partition(key.get_column(0), 0, log_num_partitions);

    constexpr gamax::size_type threshold = std::numeric_limits<gamax::size_type>::max() / 2; // TODO: be careful of overflow
    int max_work_units = num_partitions + (input_num_rows+threshold-1)/threshold;

    auto units_start_ptr = gamax::make_idx_column(max_work_units);
    auto units_end_ptr = gamax::make_idx_column(max_work_units);
    auto units_group_ptr = gamax::make_idx_column(max_work_units);

    auto units_start = units_start_ptr->data<gamax::size_type>();
    auto units_end = units_end_ptr->data<gamax::size_type>();
    auto units_group = units_group_ptr->data<gamax::size_type>();

    int num_units = 0;
    
    {
        auto partition_ps = gamax::partition::compute_partition_offsets(*partitioned_key, 0, log_num_partitions);
        gamax::groupby::generate_workunits(partition_ps->data<gamax::size_type>(), num_partitions, threshold, &num_units, units_start, units_end, units_group);
    }

    std::vector<std::unique_ptr<gamax::column>> final_columns;

    for(auto& req: agg_requests) {
        auto i = req.column_index;
        auto agg_kind = req.agg_kind;

        auto& val_col = val.get_column(i);

        auto key_type = key.get_column(0).type();

        gamax::size_type num_groups = 0;
        auto grouped_key = std::make_unique<gamax::column>(key.get_column(0).type(), estimated_num_groups, true);
        std::unique_ptr<gamax::column> res_val = nullptr;
        
        if(agg_kind == gamax::groupby::aggregation_id::SUM) {
            if(val.get_column(i).type().id() == gamax::type_id::INT32) {
                using agg_fn_t = gamax::groupby::Sum<int32_t>;
                using agg_output_t = agg_fn_t::OutputType;

                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, estimated_num_groups, true);

                auto val_data = val_col.data<agg_fn_t::InputType>();
                auto val_itr = thrust::make_permutation_iterator(val_data, idx_col->data<gamax::size_type>());

                if(key_type == gamax::data_type{gamax::type_id::INT32}) {
                    aggregate<agg_fn_t>(partitioned_key->data<int32_t>(), val_itr, 
                                    num_units, units_start, units_end, units_group, log_num_partitions, 
                                    &num_groups,
                                    grouped_key->data<int32_t>(), agg_val->data<agg_fn_t::OutputType>());
                }
                else if(key_type == gamax::data_type{gamax::type_id::INT64}) {
                    aggregate<agg_fn_t>(partitioned_key->data<int64_t>(), val_itr, 
                                    num_units, units_start, units_end, units_group, log_num_partitions, 
                                    &num_groups,
                                    grouped_key->data<int64_t>(), agg_val->data<agg_fn_t::OutputType>());
                }
                else {
                    throw std::runtime_error("Unsupported key type for SUM aggregation");
                }
                
                res_val = std::move(agg_val);
            }
            else {
                throw std::runtime_error("Unsupported value type for SUM aggregation");
            }
        }
        else if(agg_kind == gamax::groupby::aggregation_id::MAX) {
            if(val.get_column(i).type().id() == gamax::type_id::INT32) {
                using agg_fn_t = gamax::groupby::Max<int32_t>;
                using agg_output_t = agg_fn_t::OutputType;

                auto agg_val = std::make_unique<gamax::column>(gamax::data_type{type_to_id<agg_output_t>()}, estimated_num_groups, true);

                auto val_data = val_col.data<agg_fn_t::InputType>();
                auto val_itr = thrust::make_permutation_iterator(val_data, idx_col->data<gamax::size_type>());

                if(key_type == gamax::data_type{gamax::type_id::INT32}) {
                    aggregate<agg_fn_t>(partitioned_key->data<int32_t>(), val_itr, 
                                    num_units, units_start, units_end, units_group, log_num_partitions, 
                                    &num_groups,
                                    grouped_key->data<int32_t>(), agg_val->data<agg_fn_t::OutputType>());
                }
                else if(key_type == gamax::data_type{gamax::type_id::INT64}) {
                    aggregate<agg_fn_t>(partitioned_key->data<int64_t>(), val_itr, 
                                    num_units, units_start, units_end, units_group, log_num_partitions, 
                                    &num_groups,
                                    grouped_key->data<int64_t>(), agg_val->data<agg_fn_t::OutputType>());
                }
                else {
                    throw std::runtime_error("Unsupported key type for MAX aggregation");
                }
                
                res_val = std::move(agg_val);
            }
            else {
                throw std::runtime_error("Unsupported value type for MAX aggregation");
            }
        }
        else {
            throw std::runtime_error("Unsupported aggregation function");
        }

        if(final_columns.size() == 0) {
            grouped_key->set_size(num_groups);
            final_columns.push_back(std::move(grouped_key));
        }
        res_val->set_size(num_groups);
        final_columns.push_back(std::move(res_val));
    }

    cudaStreamSynchronize(0);

    return std::make_unique<gamax::table>(std::move(final_columns));
}
} // namespace gamax::groupby::partition