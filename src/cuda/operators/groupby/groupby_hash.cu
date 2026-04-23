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
#include "operators/join/materialize.hpp"
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

    // For each input row, decide its group id and record the row index
    // that became the group's "leader" (first inserter). The leader_tid
    // column, trimmed to num_groups after this kernel, is later used as
    // a gather map to materialise the output key columns — side-steps
    // writing keys inline (which fails for STRING, since STRING output
    // columns need precomputed offsets before writing).
    template<typename Set>
    __global__ void compute_groupid_kernel(
        Set set_ref,
        const gamax::size_type num_items,
        gamax::size_type* __restrict__ sparse_group_ids,
        gamax::size_type* __restrict__ group_ids_loc,
        gamax::size_type* __restrict__ leader_tid,
        gamax::size_type* __restrict__ num_groups)
    {
        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

        while (idx < num_items) {
            auto [iter, is_new_key] = set_ref.insert_and_find(idx);

            if (is_new_key) {
                gamax::size_type gid = gamax::atomic::atomic_add(num_groups, static_cast<gamax::size_type>(1));
                sparse_group_ids[idx] = gid;
                // Leader tid is indexed by gid — unique per thread in
                // the is_new_key branch, so no race.
                leader_tid[gid] = static_cast<gamax::size_type>(idx);
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

    // Helper to compute aligned buffer offset for a given aggregation index
    __device__ __forceinline__ char* get_agg_buffer_offset(
        char* base,
        gamax::size_type target_agg,
        gamax::size_type num_groups,
        gamax::experimental::mutable_table_device_view res)
    {
        char* offset = base;
        for(gamax::size_type k = 0; k < target_agg; k++) {
            std::size_t agg_size = gamax::size_of(res.get_column(k).type());
            offset += num_groups * agg_size;
            // Ensure 8-byte alignment for next aggregation buffer
            std::size_t addr = reinterpret_cast<std::size_t>(offset);
            offset = reinterpret_cast<char*>((addr + 7) & ~std::size_t(7));
        }
        return offset;
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

        // Use long long for 8-byte alignment of shared memory base
        extern __shared__ long long temp[];
        char* agg_buffer = reinterpret_cast<char*>(temp);

        // initialize aggregation buffer
        #pragma unroll
        for(gamax::size_type t = block.thread_rank(); t < num_groups; t += block.size()) {
            #pragma unroll
            for(gamax::size_type k = 0; k < num_aggs; k++) {
                char* cur_agg_buffer = get_agg_buffer_offset(agg_buffer, k, num_groups, res);
                init_agg(aggs[k].agg_kind, val.get_column(aggs[k].column_index).type(), cur_agg_buffer, t);
            }
        }
        block.sync();

        // aggregate into the shared memory buffer
        #pragma unroll
        for(gamax::size_type t = get_cuda_tid(); t < num_items; t += nthreads()) {
            const auto loc = group_ids_loc[t];
            const auto gid = sparse_group_ids[loc];

            #pragma unroll
            for(gamax::size_type i = 0; i < num_aggs; i++) {
                char* cur_agg_buffer = get_agg_buffer_offset(agg_buffer, i, num_groups, res);
                block_aggregate(
                    aggs[i].agg_kind,
                    val.get_column(aggs[i].column_index), t,
                    cur_agg_buffer, gid);
            }
        }
        block.sync();

        // Copy from shared memory to global memory
        #pragma unroll
        for(gamax::size_type t = block.thread_rank(); t < num_groups*num_aggs; t += block.size()) {
            const gamax::size_type i = t / num_groups; // which request
            const gamax::size_type j = t % num_groups; // which group
            
            char* cur_agg_buffer = get_agg_buffer_offset(agg_buffer, i, num_groups, res);
            
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

    // Multi-key groupby implementation.
    //
    // Output layout: [key_cols..., agg_cols...]. Key columns are
    // produced by gathering the input key columns through a leader-tid
    // map — one size_type per group, recording the first input row that
    // inserted each group. This side-steps writing keys inside
    // `compute_groupid_kernel` (which fails for STRING, since STRING
    // output columns require precomputed offsets). Numeric keys pay
    // the same gather cost; the benefit is a single uniform code path
    // that `simple_materialize` already dispatches correctly for STRING
    // via `gather_strings`.
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

        // Pre-allocate only the aggregation result columns — the kernels
        // need these as output buffers. Key columns are materialised
        // after we know num_groups, via a leader-tid gather; see below.
        std::vector<std::unique_ptr<gamax::column>> agg_result_cols;
        for(const auto& agg : agg_requests) {
            agg_result_cols.push_back(
                gamax::make_column(
                    gamax::groupby::agg_output_data_type(agg.agg_kind, val.get_column(agg.column_index).type()),
                    estimated_num_groups));
        }
        auto agg_result_table = std::make_unique<gamax::table>(std::move(agg_result_cols));

        constexpr double load_factor = 0.25;
        gamax::size_type constexpr empty_key_sentinel = gamax::sentinel<gamax::size_type>();
        auto cuco_allocator = *get_thrust_allocator<cuco::pair<gamax::size_type, gamax::size_type>>(stream);

        // Create device views for keys
        auto key_device_view = key.device_view(stream);

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
        // leader_tid[gid] = input-row index of the first inserter for gid.
        // Sized at estimated_num_groups (upper bound on distinct groups),
        // trimmed to num_groups post-kernel before use as a gather map.
        auto leader_tid = gamax::make_idx_column(estimated_num_groups, false);
        auto d_num_groups = gamax::make_idx_column(1, true);

        compute_groupid_kernel<<<num_tb(input_num_rows, 256), 256, 0, stream>>>(
            set_ref,
            input_num_rows,
            sparse_group_ids->data<gamax::size_type>(),
            groupid_loc->data<gamax::size_type>(),
            leader_tid->data<gamax::size_type>(),
            d_num_groups->data<gamax::size_type>());

            gamax::size_type num_groups;
            cudaMemcpyAsync(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // Aggregation side — unchanged shape, but the result table
            // now holds only agg cols (no key cols), so column indices
            // are 0..num_aggs-1 instead of num_key_cols..num_key_cols+num_aggs-1.
            std::vector<gamax::mutable_column_view> agg_result_col_views;
            for(gamax::size_type i = 0; i < agg_requests.size(); i++) {
                agg_result_col_views.push_back(agg_result_table->get_column(i).mutable_view());
            }
            gamax::mutable_table_view agg_results_view(agg_result_col_views);
            auto agg_res_device_view = agg_results_view.device_view(stream);

            auto val_view = val.device_view(stream);

            initialize_aggregation<<<num_tb(agg_result_table->num_rows(), 1024), 1024, 0, stream>>>(*val_view, *agg_res_device_view, static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()));

            // Calculate shared memory size with 8-byte alignment padding between aggregation buffers
            auto compute_aligned_smem_bytes = [&](gamax::size_type n_groups) -> std::size_t {
                std::size_t total_bytes = 0;
                for(gamax::size_type i = 0; i < agg_requests.size(); i++) {
                    total_bytes += n_groups * gamax::size_of(agg_result_table->get_column(i).type());
                    // Add alignment padding for next buffer (align to 8 bytes)
                    total_bytes = (total_bytes + 7) & ~std::size_t(7);
                }
                return total_bytes;
            };

            // Estimate bytes per group (with alignment overhead)
            std::size_t agg_bytes_per_group = compute_aligned_smem_bytes(1);

            const std::size_t max_shared_mem_size = static_cast<std::size_t>(gamax::max_smem_bytes());
            /* When the number of groups becomes larger,
            it is no longer efficient to aggregate in SMEM
            even though it goes not go beyond SMEM's capacity.
            Experimentally, we found 2048 to be the threshold */
            const auto smem_groups = std::min(max_shared_mem_size / agg_bytes_per_group, 16384UL);

            if(num_groups > smem_groups) {
                 compute_aggregation<<<num_tb(input_num_rows, 1024), 1024, 0, stream>>>(
                    *val_view,
                    *agg_res_device_view,
                    sparse_group_ids->data<gamax::size_type>(),
                    groupid_loc->data<gamax::size_type>(),
                    static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()),
                    agg_requests.size());
            } else {
                const std::size_t sm_bytes = compute_aligned_smem_bytes(num_groups);
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

            agg_result_table->set_rows(num_groups);

            // Trim leader_tid to num_groups and use it as a gather map to
            // materialise the output key columns. `simple_materialize`
            // dispatches STRING→gather_strings, numeric→util::device::gather.
            leader_tid->set_size(num_groups);
            auto leader_view = leader_tid->view();

            std::vector<std::unique_ptr<gamax::column>> final_columns;
            final_columns.reserve(num_key_cols + agg_requests.size());
            for(gamax::size_type i = 0; i < num_key_cols; i++) {
                final_columns.push_back(gamax::join::detail::simple_materialize(
                    key.get_column(i), leader_view, stream));
            }
            // release_column leaves a nullptr in place; walk forward.
            for(gamax::size_type i = 0; i < static_cast<gamax::size_type>(agg_requests.size()); i++) {
                final_columns.push_back(agg_result_table->release_column(i));
            }
            return std::make_unique<gamax::table>(std::move(final_columns));
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
