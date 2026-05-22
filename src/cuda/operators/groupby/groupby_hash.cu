#include "operators/groupby/groupby_hash.hpp"

#include <algorithm>
#include <cmath>
#include <cuco/static_set.cuh>
#include <cuda.h>
#include <memory>
#include <stdexcept>
#include <thrust/functional.h> // thrust::equal_to
#include <type_traits>

#include "compress/multikey_compress.hpp"
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

    // Book-keeping produced by the hash-set insert phase; consumed by
    // downstream aggregation or dedup materialisation.
    struct group_id_result {
        gamax::column_ptr sparse_group_ids;  // size = input rows;
                                             // set only at the leader
                                             // row of each group.
        gamax::column_ptr groupid_loc;       // size = input rows; points
                                             // at the leader row, so
                                             // sparse_group_ids[groupid_loc[i]]
                                             // is the dense group id.
        gamax::column_ptr leader_tid;        // size = num_groups (trimmed);
                                             // input-row index of the
                                             // first inserter for each
                                             // group — used as a gather
                                             // map to materialise key
                                             // columns.
        gamax::size_type num_groups;
    };

    // Build a cuco::static_set over `key`'s rows and populate the four
    // book-keeping columns above. Shared by `hash_groupby_multi_key`
    // (aggregation path) and `hash::distinct` (no-agg dedup path) so
    // the set construction and kernel launch stay in one place — the
    // 7-parameter cuco-set constructor is sensitive enough that we
    // don't want two copies drifting.
    //
    // The cuco set is released on return; downstream phases only need
    // the four outputs. Stream is synchronised before return so the
    // caller sees a fully-populated result set.
    group_id_result compute_group_ids(
        gamax::table_view key,
        gamax::groupby::groupby_hint& hint,
        cudaStream_t stream)
    {
        auto input_num_rows = key.num_rows();
        gamax::size_type estimated_num_groups =
            std::min(input_num_rows, hint.groups_upper_bound);

        constexpr double load_factor = 0.25;
        gamax::size_type constexpr empty_key_sentinel =
            gamax::sentinel<gamax::size_type>();
        auto cuco_allocator =
            *get_thrust_allocator<cuco::pair<gamax::size_type, gamax::size_type>>(stream);

        auto key_device_view = key.device_view(stream);

        auto set = cuco::static_set{
            estimated_num_groups,
            load_factor,
            cuco::empty_key{empty_key_sentinel},
            gamax::row_equal{*key_device_view},
            cuco::linear_probing<1, gamax::row_hasher>{gamax::row_hasher{*key_device_view}},
            cuco::thread_scope_device,
            cuco::storage<1>{},
            cuco_allocator,
            stream};
        auto set_ref = set.ref(cuco::insert_and_find);

        auto sparse_group_ids = gamax::make_idx_column(input_num_rows, false);
        auto groupid_loc      = gamax::make_idx_column(input_num_rows, false);
        // leader_tid is sized at the upper bound; trimmed before return
        // so it is usable as a gather map directly.
        auto leader_tid       = gamax::make_idx_column(estimated_num_groups, false);
        auto d_num_groups     = gamax::make_idx_column(1, true);

        compute_groupid_kernel<<<num_tb(input_num_rows, 256), 256, 0, stream>>>(
            set_ref,
            input_num_rows,
            sparse_group_ids->data<gamax::size_type>(),
            groupid_loc->data<gamax::size_type>(),
            leader_tid->data<gamax::size_type>(),
            d_num_groups->data<gamax::size_type>());

        gamax::size_type num_groups;
        cudaMemcpyAsync(&num_groups,
                        d_num_groups->data<gamax::size_type>(),
                        sizeof(gamax::size_type),
                        cudaMemcpyDeviceToHost,
                        stream);
        cudaStreamSynchronize(stream);

        leader_tid->set_size(num_groups);

        return {std::move(sparse_group_ids),
                std::move(groupid_loc),
                std::move(leader_tid),
                num_groups};
    }

    // Gather `key`'s columns through `leader_tid` to materialise one row
    // per distinct group. STRING is handled uniformly via
    // `simple_materialize` (dispatches to `gather_strings` / numeric
    // `util::device::gather`). Returns `num_key_cols` columns in the
    // same order as `key`.
    std::vector<std::unique_ptr<gamax::column>> gather_keys_by_leader(
        gamax::table_view key,
        gamax::column_view leader_tid,
        cudaStream_t stream)
    {
        std::vector<std::unique_ptr<gamax::column>> out;
        out.reserve(key.num_columns());
        for (gamax::size_type i = 0; i < key.num_columns(); ++i) {
            out.push_back(gamax::join::detail::simple_materialize(
                key.get_column(i), leader_tid, stream));
        }
        return out;
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

        auto gid_result = compute_group_ids(key, hint, stream);
        auto& sparse_group_ids = gid_result.sparse_group_ids;
        auto& groupid_loc      = gid_result.groupid_loc;
        auto& leader_tid       = gid_result.leader_tid;
        const auto num_groups  = gid_result.num_groups;

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
                 compute_aggregation<<<num_tb(input_num_rows, 256), 256, 0, stream>>>(
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
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, agg_fn, 256, 0);
                agg_fn<<<SM_COUNT*maxActiveBlocks, 256, sm_bytes, stream>>>(
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

            // Gather key columns through leader_tid (already trimmed to
            // num_groups by `compute_group_ids`).
            auto final_columns = gather_keys_by_leader(
                key, leader_tid->view(), stream);
            final_columns.reserve(num_key_cols + agg_requests.size());
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

// Hash-based distinct: return one row per unique combination of
// `key`'s columns, preserving column order and types. Output order
// is insertion order (first-encounter wins the group id), which
// callers that care about ordering must re-sort downstream.
//
// Shares its dedup machinery with `groupby` via `compute_group_ids` —
// STRING keys, multi-key, and empty-input behaviours carry over
// unchanged. With no aggregation requests, the path skips the
// aggregation phase entirely.
std::unique_ptr<gamax::table> distinct(
    gamax::table_view key,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    if (key.num_columns() == 0) {
        throw std::runtime_error(
            "hash::distinct: must have at least one key column");
    }
    // Empty input → empty output with the same column shape. The
    // downstream gather would trivially produce zero rows, but short-
    // circuiting here avoids launching an insert kernel over zero work.
    if (key.num_rows() == 0) {
        std::vector<std::unique_ptr<gamax::column>> empty_cols;
        empty_cols.reserve(key.num_columns());
        for (gamax::size_type i = 0; i < key.num_columns(); ++i) {
            empty_cols.push_back(gamax::make_column(
                key.get_column(i).type(), 0));
        }
        return std::make_unique<gamax::table>(std::move(empty_cols));
    }

    // Optional SKF (smart-key-fusion) path. When enabled, try to fuse
    // the multi-col key into a single packed key column. If
    // `multikey_compress` returns non-null, run the cuco-set insert on
    // the compressed single-col view — cheaper than the multi-col
    // `row_hasher`/`row_equal` — and gather the *original* multi-col
    // keys via the resulting leader_tid (output schema unchanged).
    // `multikey_compress` returns null for single-col input (nothing
    // to fuse) and when the column widths exceed the fusion bit
    // budget; either way we fall through to the default multi-col
    // path. `out_total_bits` lets us tighten the cuco-set sizing —
    // the number of distinct compressed keys is bounded by
    // 2^compressed_bits.
    gamax::column_ptr compressed_key;
    if (hint.enable_skf) {
        int compressed_bits = 0;
        gamax::compress::multikey_compress_options opts;
        opts.out_total_bits = &compressed_bits;
        compressed_key = gamax::compress::multikey_compress(key, opts, stream);
        if (compressed_key) {
            hint.groups_upper_bound = std::min(
                hint.groups_upper_bound,
                static_cast<gamax::size_type>(1) << compressed_bits);

            std::vector<gamax::column_view> compressed_cols{
                compressed_key->view()};
            gamax::table_view compressed_view(compressed_cols);

            auto gid_result = compute_group_ids(compressed_view, hint, stream);
            auto final_columns = gather_keys_by_leader(
                key, gid_result.leader_tid->view(), stream);
            return std::make_unique<gamax::table>(std::move(final_columns));
        }
        // Fall through to the multi-col path when SKF can't fuse.
    }

    auto gid_result = compute_group_ids(key, hint, stream);
    auto final_columns = gather_keys_by_leader(
        key, gid_result.leader_tid->view(), stream);
    return std::make_unique<gamax::table>(std::move(final_columns));
}
} // namespace gamax::groupby::hash
