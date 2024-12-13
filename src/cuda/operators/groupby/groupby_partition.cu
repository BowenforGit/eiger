#include "operators/groupby/groupby_partition.hpp"

#include <memory>
#include <cuda.h>
#include <type_traits>
#include <thrust/iterator/permutation_iterator.h>
#include <cuco/static_map.cuh>

#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/table_device_view.hpp"
#include "operators/partition/partition.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/load_balancer.cuh"
#include "operators/groupby/agg_func.cuh"
#include "types/data_type.hpp"
#include "agg_helper.cuh"

namespace gamax::groupby::partition {
namespace {

constexpr auto cg_size = 1;
constexpr auto window_size = 1;
constexpr auto numWindows = 4096;
auto constexpr num_windows = cuco::make_window_extent<cg_size, window_size>(cuco::extent<gamax::size_type, numWindows>{});

using namespace gamax::groupby::details;

template<typename Key, 
         std::size_t NumWindows, 
         typename IdxIt,
         int cg_size = 1,
         int window_size = 1>
__global__ void partition_groupby_aggregate_kernel(const Key* __restrict__ keys,
                                            gamax::experimental::table_device_view values, 
                                            IdxIt val_idx,
                                            const gamax::size_type num_units,
                                            gamax::size_type* __restrict__ units_start, 
                                            gamax::size_type* __restrict__ units_end, 
                                            gamax::size_type* __restrict__ num_groups,
                                            gamax::size_type* __restrict__ group_id,
                                            gamax::groupby::aggregation_request* __restrict__ aggs,
                                            const gamax::size_type num_aggs,
                                            gamax::experimental::mutable_table_device_view res) 
{
    constexpr gamax::size_type groupid_write_mask = 0x80000000;
    constexpr gamax::size_type groupid_read_mask = 0x7FFFFFFF;

    using Value     = gamax::size_type; // tuple id
    using slot_type = cuco::pair<Key, Value>;
    using extent_type      = cuco::extent<std::size_t, NumWindows>;
    using storage_ref_type = cuco::aow_storage_ref<slot_type, window_size, extent_type>;

    extern __shared__ int16_t temp[];
    auto map = reinterpret_cast<cuco::window<slot_type, window_size>*>(temp);
    struct {
        gamax::size_type num_groups_local;
        gamax::size_type block_offset;
    } *u = reinterpret_cast<decltype(u)>(map+NumWindows);
    
    auto group_loc = group_id;

    auto key_arr = keys;

    auto const block = cooperative_groups::this_thread_block();
    
    for(gamax::size_type i = blockIdx.x; i < num_units; i += gridDim.x) {
        if(block.thread_rank() == 0) u->num_groups_local = 0;
        const auto ustart = units_start[i];
        const auto uend = units_end[i];

        auto raw_ref =
            cuco::static_map_ref{cuco::empty_key<Key>{-1}, // TODO: reconsider the sentinel value
                                cuco::empty_value<Value>{-1},
                                thrust::equal_to<Key>{},
                                cuco::linear_probing<cg_size, cuco::default_hash_function<Key>>{}, // the hash computation is actually not a bottleneck here
                                cuco::thread_scope_block,
                                storage_ref_type{extent_type{}, map}};
        raw_ref.initialize(block);

        auto insert_and_find_ref = std::move(raw_ref).with(cuco::op::insert_and_find);

        #pragma unroll
        for(gamax::size_type t = block.thread_rank(); t < uend - ustart; t += block.size()) {
            auto idx = t+ustart;
            auto [slot, is_new_key] = insert_and_find_ref.insert_and_find(cuco::pair{key_arr[idx], idx});
            if(is_new_key) {
                auto t_group_id = atomicAdd(&(u->num_groups_local), 1); // block-scope group id
                group_id[idx] = (groupid_write_mask | t_group_id); // flag that his is a group id not a location
            } else {
                group_loc[idx] = slot->second;
            }
        }
        block.sync();

        if(block.thread_rank() == 0) {
            u->block_offset = atomicAdd(num_groups, u->num_groups_local);
        }
        block.sync();

        auto block_offset = u->block_offset;
        auto num_groups_local = u->num_groups_local;
        block.sync();


        if(num_groups_local == 1) {
            if(block.thread_rank() == 0) {
                res[0].data<Key>()[block_offset] = key_arr[ustart];
            }
            for(gamax::size_type k = 0; k < num_aggs; k++) {
                gamax::groupby::details::reduce(
                    block,
                    aggs[k].agg_kind,
                    values.get_column(aggs[k].column_index), 
                    val_idx, ustart, uend,
                    res.get_column(k+1), block_offset);
            }
        } else if(num_groups_local <= 256) { // TODO: re-evaluate this condition
            // void* agg_buffer = reinterpret_cast<void*>(u+1);
            void* agg_buffer = reinterpret_cast<void*>(temp);

            // initialize the aggregation buffer in SMEM
            auto cur_agg_buffer = agg_buffer;

            #pragma unroll
            for(gamax::size_type t = block.thread_rank(); t < num_groups_local; t += block.size()) {
                cur_agg_buffer = agg_buffer;
                
                #pragma unroll
                for(gamax::size_type k = 0; k < num_aggs; k++) {
                    init_agg(aggs[k].agg_kind, values.get_column(k).type(), cur_agg_buffer, t);
                    cur_agg_buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(cur_agg_buffer) + (num_groups_local)*gamax::size_of(res.get_column(k+1).type()));
                }
            }
            block.sync();
            
            #pragma unroll
            for(gamax::size_type t = block.thread_rank(); t < uend - ustart; t += block.size()) {
                const auto idx = t+ustart;

                gamax::size_type l_group_id {-1}; // block-local group id
                auto gl = group_loc[idx];
                if((gl & groupid_read_mask) != gl) { // the first inserter
                    l_group_id = (gl & groupid_read_mask);
                    res[0].data<Key>()[block_offset+l_group_id] = key_arr[idx];
                } else {
                    l_group_id = (group_id[gl] & groupid_read_mask);
                }
                
                auto cur_agg_buffer = agg_buffer;
                
                #pragma unroll
                for(gamax::size_type k = 0; k < num_aggs; k++) {
                    block_aggregate(
                        aggs[k].agg_kind,
                        values.get_column(aggs[k].column_index), val_idx[idx],
                        cur_agg_buffer, l_group_id);
                    cur_agg_buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(cur_agg_buffer) + (num_groups_local)*gamax::size_of(res.get_column(k+1).type()));
                }
            }
            block.sync();
            
            // copy the result
            copy_agg_from_smem_to_global(
                block,
                res,
                block_offset,
                agg_buffer,
                num_groups_local,
                num_aggs);
        } else {
            for(gamax::size_type t = block.thread_rank(); t < uend - ustart; t += block.size()) {
                auto idx = t+ustart;

                gamax::size_type g_group_id {-1}; // global group id
                auto gl = group_loc[idx];
                if((gl & groupid_read_mask) != gl) { // the first inserter
                    g_group_id = block_offset + (gl & groupid_read_mask);
                    res[0].data<Key>()[g_group_id] = key_arr[idx];
                } else {
                    g_group_id = block_offset + (group_id[gl] & groupid_read_mask);
                }

                #pragma unroll
                for(gamax::size_type k = 0; k < num_aggs; k++) {
                    gamax::groupby::details::aggregate(
                        aggs[k].agg_kind,
                        values.get_column(aggs[k].column_index), val_idx[idx],
                        res.get_column(k+1), g_group_id);
                }
            }
        }
        block.sync();
    }
}

template<typename Key, int window_size, int numWindows, typename Itr>
std::unique_ptr<gamax::table> compute_aggregate(
    gamax::column_view partitioned_key,
    const gamax::size_type estimated_num_groups,
    const gamax::size_type log_num_partitions,
    gamax::table_view val,
    Itr itr,
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    cudaStream_t stream)
{
    auto input_num_rows = partitioned_key.size();
    const gamax::size_type num_partitions = (1 << log_num_partitions);

    constexpr gamax::size_type threshold = std::numeric_limits<gamax::size_type>::max() / 2; // TODO: be careful of overflow
    auto max_work_units = num_partitions + (input_num_rows+threshold-1)/threshold;

    auto units_start_ptr = gamax::make_idx_column(max_work_units);
    auto units_end_ptr = gamax::make_idx_column(max_work_units);
    auto units_group_ptr = gamax::make_idx_column(max_work_units);

    gamax::size_type num_units = 0;
    auto partition_ps = gamax::partition::compute_partition_offsets(partitioned_key, 0, log_num_partitions);
    gamax::groupby::generate_workunits(
        partition_ps->data<gamax::size_type>(), 
        num_partitions, 
        threshold, 
        &num_units, 
        units_start_ptr->data<gamax::size_type>(), 
        units_end_ptr->data<gamax::size_type>(), 
        units_group_ptr->data<gamax::size_type>(),
        stream);
    partition_ps.reset();

    std::vector<std::unique_ptr<gamax::column>> final_columns;
    final_columns.push_back(
        gamax::make_column(
            partitioned_key.type(), 
            estimated_num_groups));
    for(const auto& agg : agg_requests) {
        final_columns.push_back(
            gamax::make_column(
                gamax::groupby::agg_output_data_type(agg.agg_kind, val.get_column(agg.column_index).type()), 
                estimated_num_groups));
    }
    auto res = std::make_unique<gamax::table>(std::move(final_columns));
    // auto res_view = table_device_view::create(*res, stream);
    auto res_view = res->mutable_view().device_view(stream);
    
    auto d_agg_req_buf = std::make_unique<gamax::device_buffer>(sizeof(gamax::groupby::aggregation_request) * agg_requests.size(), false);
    cudaMemcpy(d_agg_req_buf->data(), agg_requests.data(), d_agg_req_buf->size(), cudaMemcpyHostToDevice);

    // auto val_view = table_device_view::create(val, stream);
    auto val_view = val.device_view(stream);
    initialize_aggregation<<<num_tb(res->num_rows(), 1024), 1024, 0, stream>>>(*val_view, *res_view, static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()));
    
    std::size_t sm_bytes = numWindows * sizeof(cuco::window<cuco::pair<Key, gamax::size_type>, window_size>) + sizeof(gamax::size_type)*2;
    if(estimated_num_groups / num_partitions <= 256) {
        std::size_t agg_bytes = 0;
        for(gamax::size_type i = 1; i < res->num_columns(); i++) {
            agg_bytes += gamax::size_of(res->get_column(i).type());
        }
        sm_bytes = std::max(agg_bytes * 256, sm_bytes);
    }
    // std::cout << "sm_bytes: " << sm_bytes << std::endl; 
    auto agg_fn = partition_groupby_aggregate_kernel<Key, numWindows, decltype(itr)>;
    if(sm_bytes > 48*1024) {
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(agg_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes));
    }

    auto group_id = gamax::make_idx_column(input_num_rows, false);
    auto d_num_groups = gamax::make_idx_column(1, true);
    // the grid size and block size given by cudaOccupancyMaxPotentialBlockSize 
    // do not work very well for this kernel
    agg_fn<<<num_units, 768, sm_bytes, stream>>>(
        partitioned_key.data<Key>(),
        *val_view,
        itr,
        num_units,
        units_start_ptr->data<gamax::size_type>(),
        units_end_ptr->data<gamax::size_type>(),
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

template<int window_size, int numWindows, bool kUseGFUR = false>
struct partition_groupby_impl {
    template<typename Key>
    std::unique_ptr<gamax::table> operator()(
            gamax::table_view key,
            gamax::table_view val,
            std::vector<gamax::groupby::aggregation_request>& agg_requests,
            gamax::groupby::groupby_hint& hint,
            cudaStream_t stream = cudaStreamDefault)
    {
        gamax::size_type estimated_num_groups = std::min(key.num_rows(), hint.groups_upper_bound);
        
        // int log_num_partitions = (estimated_num_groups * 8 / (1 << 8) > numWindows) ? 16 : 8;
        int log_num_partitions = 16;

        if constexpr (kUseGFUR) {
            auto [partitioned_key, permutation_idx] = gamax::partition::arg_partition(key.get_column(0), 0, log_num_partitions, nullptr, 0, stream);

            return compute_aggregate<Key, window_size, numWindows>(
                partitioned_key->view(), 
                estimated_num_groups, 
                log_num_partitions, 
                val, 
                permutation_idx->data<gamax::size_type>(),
                agg_requests, 
                stream);
        } else {
            std::unique_ptr<gamax::column> partitioned_key {nullptr};
        
            std::vector<std::unique_ptr<gamax::column>> partitioned_val_cols;
            std::unique_ptr<gamax::column> pval_temp {nullptr};
            for(gamax::size_type i = 0; i < val.num_columns(); i++) {
                std::tie(partitioned_key, pval_temp) = gamax::partition::partition_column_pair(key.get_column(0), val.get_column(i), 0, log_num_partitions, stream);
                partitioned_val_cols.push_back(std::move(pval_temp));
            }
            auto partitioned_val = std::make_unique<gamax::table>(std::move(partitioned_val_cols));
            
            return compute_aggregate<Key, window_size, numWindows>(
                partitioned_key->view(), 
                estimated_num_groups, 
                log_num_partitions, 
                partitioned_val->view(), 
                thrust::make_counting_iterator<gamax::size_type>(0),
                agg_requests, 
                stream);
        }
    }
    };

} // unnamed namespace

std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{   
    return type_dispatcher(
        key.get_column(0).type(),
        partition_groupby_impl<window_size, num_windows.value(), false>{},
        key, val, agg_requests, hint, stream);
}

std::unique_ptr<gamax::table> groupby_gfur(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{   
    return type_dispatcher(
        key.get_column(0).type(),
        partition_groupby_impl<window_size, num_windows.value(), true>{},
        key, val, agg_requests, hint, stream);
}
} // namespace gamax::groupby::partition