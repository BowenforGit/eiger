#include "operators/groupby/groupby_hash.hpp"

#include <cmath>
#include <cuco/static_set.cuh>
#include <cuda.h>
#include <memory>
#include <stdexcept>
#include <thrust/functional.h> // thrust::equal_to
#include <type_traits>

#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/row_utils.cuh"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "types/data_type.hpp"
#include "types/sentinel.hpp"
#include "agg_helper.cuh"
#include "atomics/atomics.cuh"
#include <utils/device_info.hpp>


namespace gamax::groupby::hash {
namespace {
    using namespace gamax::groupby::details;

    void check_solvable(gamax::table_view key, gamax::table_view val, std::vector<gamax::groupby::aggregation_request>& agg_requests) {
        if(key.num_columns() == 0) {
            throw std::runtime_error("Aggregation only is not supported yet");
        }

        if(agg_requests.size() == 0) {
            throw std::runtime_error("No aggregation function is specified");
        }
    }

    template<typename Set>
    __global__ void compute_groupid_kernel(
        Set set_ref,
        gamax::experimental::table_device_view keys_in,
        gamax::experimental::mutable_table_device_view keys_out,
        const gamax::size_type num_items,
        gamax::size_type* __restrict__ sparse_group_ids,
        gamax::size_type* __restrict__ group_ids_loc,
        gamax::size_type* __restrict__ num_groups)
    {
        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

        while (idx < num_items) {
            auto [iter, is_new_key] = set_ref.insert_and_find(idx);
            
            if (is_new_key) {
                gamax::size_type gid = gamax::atomic::atomic_add(num_groups, static_cast<gamax::size_type>(1));
                sparse_group_ids[idx] = gid;
                // Copy all key columns to output
                gamax::copy_row(keys_in, keys_out, idx, gid);
            } 

            group_ids_loc[idx] = *iter;
            
            idx += loop_stride;
        }
    }

    __global__ void compute_aggregation(
        gamax::experimental::table_device_view val,
        gamax::experimental::mutable_table_device_view res,
        const gamax::size_type* __restrict__ sparse_group_ids,
        const gamax::size_type* __restrict__ group_ids_loc,
        const gamax::groupby::aggregation_request* __restrict__ aggs,
        const gamax::size_type num_aggs) 
    {
        auto num_items = val.num_rows();

        #pragma unroll
        for(gamax::size_type t = get_cuda_tid(); t < num_items; t += nthreads()) {
            const auto loc = group_ids_loc[t];
            const auto gid = sparse_group_ids[loc];
            #pragma unroll
            for(gamax::size_type i = 0; i < num_aggs; i++) {
                gamax::groupby::details::aggregate(
                    aggs[i].agg_kind,
                    val.get_column(aggs[i].column_index), t,
                    res.get_column(i), gid);
            }
        }
    }

    __global__ void compute_aggregation_block(
        gamax::experimental::table_device_view val,
        gamax::experimental::mutable_table_device_view res,
        const gamax::size_type num_groups,
        const gamax::size_type* __restrict__ sparse_group_ids,
        const gamax::size_type* __restrict__ group_ids_loc,
        const gamax::groupby::aggregation_request* __restrict__ aggs,
        const gamax::size_type num_aggs) 
    {
        const auto num_items = val.num_rows();

        auto const block = cooperative_groups::this_thread_block();

        extern __shared__ int16_t temp[];
        char* agg_buffer = reinterpret_cast<char*>(temp);

        // initialize aggregation buffer
        char* cur_agg_buffer;
        
        #pragma unroll
        for(gamax::size_type t = block.thread_rank(); t < num_groups; t += block.size()) {
            cur_agg_buffer = agg_buffer;

            #pragma unroll
            for(gamax::size_type k = 0; k < num_aggs; k++) {
                init_agg(aggs[k].agg_kind, val.get_column(aggs[k].column_index).type(), cur_agg_buffer, t);
                cur_agg_buffer = cur_agg_buffer + (num_groups)*gamax::size_of(res.get_column(k).type());
            }
        }
        block.sync();

        // aggregate into the shared memory buffer
        #pragma unroll
        for(gamax::size_type t = get_cuda_tid(); t < num_items; t += nthreads()) {
            const auto loc = group_ids_loc[t];
            const auto gid = sparse_group_ids[loc];

            cur_agg_buffer = agg_buffer;

            #pragma unroll
            for(gamax::size_type i = 0; i < num_aggs; i++) {
                block_aggregate(
                    aggs[i].agg_kind,
                    val.get_column(aggs[i].column_index), t,
                    cur_agg_buffer, gid);
                cur_agg_buffer = cur_agg_buffer + (num_groups)*gamax::size_of(res.get_column(i).type());
            }
        }
        block.sync();

        cur_agg_buffer = agg_buffer;

        gamax::size_type last_i = 0;
        
        #pragma unroll
        for(gamax::size_type t = block.thread_rank(); t < num_groups*num_aggs; t += block.size()) {
            const gamax::size_type i = t / num_groups; // which request
            const gamax::size_type j = t % num_groups; // which group
            
            #pragma unroll
            for(gamax::size_type k = last_i; k < i; k++) {
                cur_agg_buffer = cur_agg_buffer + (num_groups)*gamax::size_of(res.get_column(k).type());
            }
            last_i = i;
            
            auto out_type = res.get_column(i).type();
            gamax::column_view temp_col(cur_agg_buffer, 
                                        num_groups, 
                                        out_type);
            gamax::groupby::details::aggregate(
                aggs[i].agg_kind,
                temp_col, j,
                res.get_column(i), j);
        }
    }

    // Multi-key groupby implementation
    std::unique_ptr<gamax::table> hash_groupby_multi_key(
        gamax::table_view key,
        gamax::table_view val,
        std::vector<gamax::groupby::aggregation_request>& agg_requests,
        gamax::groupby::groupby_hint& hint,
        cudaStream_t stream = cudaStreamDefault)
    {
        auto input_num_rows = key.num_rows();
        auto num_key_cols = key.num_columns();
        gamax::size_type estimated_num_groups = std::min(input_num_rows, hint.groups_upper_bound);

        auto d_agg_req_buf = std::make_unique<gamax::device_buffer>(sizeof(gamax::groupby::aggregation_request) * agg_requests.size(), false);
        cudaMemcpy(d_agg_req_buf->data(), agg_requests.data(), d_agg_req_buf->size(), cudaMemcpyHostToDevice);
        
        // Create result columns for ALL key columns first, then aggregation results
        std::vector<std::unique_ptr<gamax::column>> final_columns;
        for(gamax::size_type i = 0; i < num_key_cols; i++) {
            final_columns.push_back(
                gamax::make_column(
                    key.get_column(i).type(), 
                    estimated_num_groups, 
                    false));
        }
        for(const auto& agg : agg_requests) {
            final_columns.push_back(
                gamax::make_column(
                    gamax::groupby::agg_output_data_type(agg.agg_kind, val.get_column(agg.column_index).type()), 
                    estimated_num_groups));
        }
        auto res = std::make_unique<gamax::table>(std::move(final_columns));
        
        constexpr double load_factor = 0.25;
        gamax::size_type constexpr empty_key_sentinel = gamax::sentinel<gamax::size_type>();
        auto cuco_allocator = *get_thrust_allocator<cuco::pair<gamax::size_type, gamax::size_type>>(stream);   

        // Create device views for keys
        auto key_device_view = key.device_view(stream);
        
        // Build output keys table view (only key columns from result)
        std::vector<gamax::mutable_column_view> res_key_col_views;
        for(gamax::size_type i = 0; i < num_key_cols; i++) {
            res_key_col_views.push_back(res->get_column(i).mutable_view());
        }
        gamax::mutable_table_view res_keys_view(res_key_col_views);
        auto res_keys_device_view = res_keys_view.device_view(stream);

        // Create the hash set using row-based equality and hashing
        auto set = cuco::static_set{estimated_num_groups,
                                load_factor,
                                cuco::empty_key{empty_key_sentinel},
                                gamax::row_equal{*key_device_view},
                                cuco::linear_probing<1, gamax::row_hasher>{gamax::row_hasher{*key_device_view}},
                                cuco::thread_scope_device,
                                cuco::storage<1>{},
                                cuco_allocator, 
                                stream
                                };
        auto set_ref = set.ref(cuco::insert_and_find);

        auto sparse_group_ids = gamax::make_idx_column(input_num_rows, false);
        auto groupid_loc = gamax::make_idx_column(input_num_rows, false);
        auto d_num_groups = gamax::make_idx_column(1, true);

        compute_groupid_kernel<<<num_tb(input_num_rows, 256), 256, 0, stream>>>(
            set_ref, 
            *key_device_view, 
            *res_keys_device_view,
            input_num_rows, 
            sparse_group_ids->data<gamax::size_type>(), 
            groupid_loc->data<gamax::size_type>(), 
            d_num_groups->data<gamax::size_type>());

            gamax::size_type num_groups;
            cudaMemcpyAsync(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // Create a view of only the aggregation result columns (after key columns)
            std::vector<gamax::mutable_column_view> agg_result_col_views;
            for(gamax::size_type i = 0; i < agg_requests.size(); i++) {
                agg_result_col_views.push_back(res->get_column(num_key_cols + i).mutable_view());
            }
            gamax::mutable_table_view agg_results_view(agg_result_col_views);
            auto agg_res_device_view = agg_results_view.device_view(stream);

            auto val_view = val.device_view(stream);

            initialize_aggregation<<<num_tb(res->num_rows(), 1024), 1024, 0, stream>>>(*val_view, *agg_res_device_view, static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()));
            
            std::size_t agg_bytes = 0;
            for(gamax::size_type i = 0; i < agg_requests.size(); i++) {
                agg_bytes += gamax::size_of(res->get_column(num_key_cols + i).type());
            }
            
            const std::size_t max_shared_mem_size = static_cast<std::size_t>(gamax::max_smem_bytes());
            /* When the number of groups becomes larger, 
            it is no longer efficient to aggregate in SMEM 
            even though it goes not go beyond SMEM's capacity.
            Experimentally, we found 2048 to be the threshold */
            const auto smem_groups = std::min(max_shared_mem_size / agg_bytes, 16384UL);

            if(num_groups > smem_groups) {
                 compute_aggregation<<<num_tb(input_num_rows, 1024), 1024, 0, stream>>>(
                    *val_view, 
                    *agg_res_device_view, 
                    sparse_group_ids->data<gamax::size_type>(), 
                    groupid_loc->data<gamax::size_type>(), 
                    static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()), 
                    agg_requests.size());
            } else {
                const std::size_t sm_bytes = num_groups * agg_bytes;
                auto agg_fn = compute_aggregation_block;
#ifdef __NVCC__ // only works for CUDA because AMD will ignore
                if(sm_bytes > 48*1024) {
                    CHECK_CUDA_ERROR(cudaFuncSetAttribute(compute_aggregation_block, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes));
                }
#endif
                int maxActiveBlocks;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, agg_fn, 1024, 0);
                agg_fn<<<SM_COUNT*maxActiveBlocks, 1024, sm_bytes, stream>>>(
                    *val_view, 
                    *agg_res_device_view, 
                    num_groups,
                    sparse_group_ids->data<gamax::size_type>(), 
                    groupid_loc->data<gamax::size_type>(), 
                    static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()), 
                    agg_requests.size());
                CHECK_LAST_CUDA_ERROR();
            }

            cudaStreamSynchronize(stream);

            res->set_rows(num_groups);
            return res;
    }
} // unnamed namespace

// Hash-based groupby supporting multiple key columns
std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    check_solvable(key, val, agg_requests);
    
    return hash_groupby_multi_key(key, val, agg_requests, hint, stream);
}
} // namespace gamax::groupby::hash
