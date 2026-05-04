#include "operators/groupby/groupby_partition.hpp"

#include <memory>
#include <cuda.h>
#include <type_traits>
#include <thrust/iterator/permutation_iterator.h>
#include <cuco/static_set.cuh>
#include <cuco/bucket_storage.cuh>

#include "utils/utils.cuh"
#include "utils/transform.hpp"
#include "atomics/atomics.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/table_device_view.hpp"
#include "operators/partition/partition.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/load_balancer.cuh"
#include "operators/groupby/agg_func.cuh"
#include "operators/sort/sort.hpp"
#include "types/data_type.hpp"
#include "types/sentinel.hpp"
#include "agg_helper.cuh"

namespace gamax::groupby::partition {
namespace {

constexpr auto cg_size = 1;
constexpr auto bucket_size = 1;
constexpr std::size_t num_slots = 4096;

using namespace gamax::groupby::details;

// Custom hasher for index-based static_set: hashes the actual key value at the given index
template<typename Key>
struct partition_key_hasher {
    const Key* keys;
    
    __device__ auto operator()(gamax::size_type idx) const {
        return cuco::default_hash_function<Key>{}(keys[idx]);
    }
};

// Custom equality for index-based static_set: compares actual key values at given indices
template<typename Key>
struct partition_key_equal {
    const Key* keys;

    __device__ bool operator()(gamax::size_type lhs, gamax::size_type rhs) const {
        return keys[lhs] == keys[rhs];
    }
};

// MurmurHash3 fmix64 / fmix32 finalizers. Bijective on the full output width
// (every constant is odd, every shift is self-invertible), so distinct keys
// produce distinct hashes; collisions only arise from the subsequent low-bit
// mask. Used to spread low-entropy keys (e.g. TPC-H orderkeys, multiples of 4)
// uniformly across partitions; without this the radix partition leaves a large
// fraction of buckets empty and overloads the rest.
__host__ __device__ __forceinline__ std::uint64_t fmix64(std::uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__host__ __device__ __forceinline__ std::uint32_t fmix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6bU;
    x ^= x >> 13;
    x *= 0xc2b2ae35U;
    x ^= x >> 16;
    return x;
}

template<typename Key>
__host__ __device__ __forceinline__ std::uint64_t bit_mix(Key k) {
    if constexpr (sizeof(Key) <= 4) {
        return static_cast<std::uint64_t>(fmix32(static_cast<std::uint32_t>(k)));
    } else {
        return fmix64(static_cast<std::uint64_t>(k));
    }
}

template<typename Key>
__global__ void compute_partition_ids_kernel(
    const Key* __restrict__ keys,
    gamax::size_type* __restrict__ pids,
    gamax::size_type n,
    std::uint64_t mask)
{
    for(gamax::size_type t = get_cuda_tid(); t < n; t += nthreads()) {
        pids[t] = static_cast<gamax::size_type>(bit_mix(keys[t]) & mask);
    }
}

// Per-block state that lives in shared memory immediately after the slot table.
// Hoisted to a named struct so the host can size shared memory via sizeof().
template<typename Key>
struct partition_block_state {
    gamax::size_type num_groups_local;
    gamax::size_type block_offset;
    const Key* key_ptr;       // Key pointer cached in SMEM for the static_set functors.
    int local_overflow;       // Set when this partition's distinct keys saturate the slot table.
};

template<typename Key, 
         std::size_t NumSlots, 
         typename IdxIt,
         int CgSize = 1,
         int BucketSize = 1>
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
                                            gamax::experimental::mutable_table_device_view res,
                                            int* __restrict__ overflow_flag)
{
    constexpr gamax::size_type groupid_write_mask = 0x80000000;
    constexpr gamax::size_type groupid_read_mask = 0x7FFFFFFF;

    // Now we store row indices instead of (Key, Value) pairs - halves slot memory footprint
    using slot_type = gamax::size_type;
    using extent_type      = cuco::extent<std::size_t, NumSlots>;
    using storage_ref_type = cuco::bucket_storage_ref<slot_type, BucketSize, extent_type>;

    // Use long long for 8-byte alignment of shared memory base
    extern __shared__ long long temp[];
    auto shared_slots = reinterpret_cast<slot_type*>(temp);
    auto u = reinterpret_cast<partition_block_state<Key>*>(shared_slots + NumSlots);
    
    auto group_loc = group_id;

    auto key_arr = keys;

    auto const block = cooperative_groups::this_thread_block();
    
    for(gamax::size_type i = blockIdx.x; i < num_units; i += gridDim.x) {
        if(block.thread_rank() == 0) {
            u->num_groups_local = 0;
            u->key_ptr = key_arr;  // Store key pointer in shared memory
            u->local_overflow = 0;
        }
        block.sync();  // Ensure key_ptr is visible to all threads
        
        const auto ustart = units_start[i];
        const auto uend = units_end[i];

        // Create static_set with index-based hashing and equality using key pointer from shared memory
        auto raw_ref =
            cuco::static_set_ref{cuco::empty_key<gamax::size_type>{gamax::sentinel<gamax::size_type>()},
                                partition_key_equal<Key>{u->key_ptr},
                                cuco::linear_probing<CgSize, partition_key_hasher<Key>>{partition_key_hasher<Key>{u->key_ptr}},
                                cuco::cuda_thread_scope<cuda::thread_scope_block>{},
                                storage_ref_type{extent_type{}, shared_slots}};
        raw_ref.initialize(block);

        auto insert_and_find_ref = raw_ref.rebind_operators(cuco::op::insert_and_find);

        #pragma unroll
        for(gamax::size_type t = block.thread_rank(); t < uend - ustart; t += block.size()) {
            auto idx = t+ustart;
            // Insert the row index; the custom hasher/equal handle key comparison via indirection
            auto [iter, is_new_key] = insert_and_find_ref.insert_and_find(idx);
            if(is_new_key) {
                auto t_group_id = gamax::atomic::atomic_add(&(u->num_groups_local), gamax::size_type{1}); // block-scope group id
                group_id[idx] = (groupid_write_mask | t_group_id); // flag that this is a group id not a location
            } else {
                group_loc[idx] = *iter;  // Dereference iterator to get the representative row index
            }
        }
        block.sync();

        // Slot-table saturation detection: cuco::static_set with linear probing
        // requires at least one empty slot for correct lookups. Once num_groups_local
        // reaches NumSlots, subsequent new-key inserts may silently alias existing
        // slots and produce phantom groups. Flag and skip the result-write phase;
        // the host will throw after the kernel returns.
        if(block.thread_rank() == 0 && u->num_groups_local >= static_cast<gamax::size_type>(NumSlots)) {
            u->local_overflow = 1;
            atomicExch(overflow_flag, 1);
        }
        block.sync();
        if(u->local_overflow != 0) {
            block.sync();
            continue;
        }

        if(block.thread_rank() == 0) {
            u->block_offset = gamax::atomic::atomic_add(num_groups, u->num_groups_local);
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
            char* agg_buffer = reinterpret_cast<char*>(temp);

            // initialize the aggregation buffer in SMEM
            #pragma unroll
            for(gamax::size_type t = block.thread_rank(); t < num_groups_local; t += block.size()) {
                #pragma unroll
                for(gamax::size_type k = 0; k < num_aggs; k++) {
                    char* cur_agg_buffer = get_partition_agg_buffer_offset(agg_buffer, k, num_groups_local, res);
                    init_agg(aggs[k].agg_kind, values.get_column(aggs[k].column_index).type(), cur_agg_buffer, t);
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
                
                #pragma unroll
                for(gamax::size_type k = 0; k < num_aggs; k++) {
                    char* cur_agg_buffer = get_partition_agg_buffer_offset(agg_buffer, k, num_groups_local, res);
                    block_aggregate(
                        aggs[k].agg_kind,
                        values.get_column(aggs[k].column_index), val_idx[idx],
                        cur_agg_buffer, l_group_id);
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

template<typename Key, int BucketSize, std::size_t NumSlots, typename Itr>
std::unique_ptr<gamax::table> compute_aggregate(
    gamax::column_view partitioned_key,
    std::unique_ptr<gamax::column> partition_ps,
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
    
    // Create a view of only the aggregation result columns (after key column)
    std::vector<gamax::mutable_column_view> agg_result_col_views;
    for(gamax::size_type i = 0; i < agg_requests.size(); i++) {
        agg_result_col_views.push_back(res->get_column(1 + i).mutable_view());
    }
    gamax::mutable_table_view agg_results_view(agg_result_col_views);
    auto agg_res_device_view = agg_results_view.device_view(stream);
    
    auto res_view = res->mutable_view().device_view(stream);
    
    auto d_agg_req_buf = std::make_unique<gamax::device_buffer>(sizeof(gamax::groupby::aggregation_request) * agg_requests.size(), false);
    cudaMemcpyAsync(d_agg_req_buf->data(), agg_requests.data(), d_agg_req_buf->size(), cudaMemcpyHostToDevice, stream);

    // auto val_view = table_device_view::create(val, stream);
    auto val_view = val.device_view(stream);
    initialize_aggregation<<<num_tb(res->num_rows(), 1024), 1024, 0, stream>>>(*val_view, *agg_res_device_view, static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()));
    
    // Shared memory layout: NumSlots size_type slots (row indices), followed by
    // the per-block bookkeeping struct. Use sizeof(partition_block_state<Key>)
    // so additions to the struct (e.g. local_overflow) don't get under-budgeted.
    std::size_t sm_bytes = num_slots * sizeof(gamax::size_type) + sizeof(partition_block_state<Key>);
    if(estimated_num_groups / num_partitions <= 256) {
        // Calculate aligned shared memory size for aggregation buffers
        std::size_t agg_bytes = 0;
        for(gamax::size_type i = 1; i < res->num_columns(); i++) {
            agg_bytes += 256 * gamax::size_of(res->get_column(i).type());
            // Add alignment padding (align to 8 bytes)
            agg_bytes = (agg_bytes + 7) & ~std::size_t(7);
        }
        sm_bytes = std::max(agg_bytes, sm_bytes);
    }
    // std::cout << "sm_bytes: " << sm_bytes << std::endl; 
    auto agg_fn = partition_groupby_aggregate_kernel<Key, NumSlots, decltype(itr)>;
    if(sm_bytes > 48*1024) {
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(agg_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes));
    }

    auto group_id = gamax::make_idx_column(input_num_rows, false, stream);
    auto d_num_groups = gamax::make_idx_column(1, true, stream);
    auto d_overflow_flag = std::make_unique<gamax::device_buffer>(sizeof(int), false);
    cudaMemsetAsync(d_overflow_flag->data(), 0, sizeof(int), stream);
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
        *res_view,
        static_cast<int*>(d_overflow_flag->data()));

    gamax::size_type num_groups;
    int h_overflow = 0;
    cudaMemcpyAsync(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_overflow, d_overflow_flag->data(), sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    if(h_overflow != 0) {
        throw std::runtime_error(
            "partition groupby: per-partition slot table overflow (NumSlots=" +
            std::to_string(NumSlots) +
            "). Distinct keys in some partition exceeded the SMEM hash set capacity. "
            "Increase log_num_partitions or route to HASH groupby.");
    }

    res->set_rows(num_groups);

    return res;
}

template<int BucketSize, std::size_t NumSlots, bool kUseGFUR = false>
struct partition_groupby_impl {
    template<typename Key, typename std::enable_if_t<!std::is_integral<Key>::value, int> = 0>
    std::unique_ptr<gamax::table> operator()(
            gamax::table_view key,
            gamax::table_view val,
            std::vector<gamax::groupby::aggregation_request>& agg_requests,
            gamax::groupby::groupby_hint& hint,
            cudaStream_t stream = cudaStreamDefault)
    {
        throw std::runtime_error("Non-integral key type is not supported for partition-based groupby.");
    }

    template<typename Key, typename std::enable_if_t<std::is_integral<Key>::value, int> = 0>
    std::unique_ptr<gamax::table> operator()(
            gamax::table_view key,
            gamax::table_view val,
            std::vector<gamax::groupby::aggregation_request>& agg_requests,
            gamax::groupby::groupby_hint& hint,
            cudaStream_t stream = cudaStreamDefault)
    {
        gamax::size_type estimated_num_groups = std::min(key.num_rows(), hint.groups_upper_bound);

        // int log_num_partitions = (estimated_num_groups * 8 / (1 << 8) > NumSlots) ? 16 : 8;
        int log_num_partitions = 16;

        const auto input_num_rows = key.num_rows();
        const std::uint64_t mask = (std::uint64_t{1} << log_num_partitions) - 1;

        // Compute per-row partition_id = bit_mix(key) & mask. Bit-mixing the key
        // before extracting low radix bits prevents skew when the key has low
        // entropy in its low bits (e.g. TPC-H orderkeys are multiples of 4, so a
        // plain low-bit radix would empty 3/4 of the partitions and overload the
        // remaining 1/4 past the per-block slot table capacity).
        auto pid_col = gamax::make_idx_column(input_num_rows, false, stream);
        {
            const auto threads = 1024;
            const auto blocks = num_tb(input_num_rows, threads);
            compute_partition_ids_kernel<Key><<<blocks, threads, 0, stream>>>(
                key.get_column(0).data<Key>(),
                pid_col->data<gamax::size_type>(),
                input_num_rows,
                mask);
        }

        // Multi-way partition: a single CUB radix pass over log_num_partitions
        // bits of pid (cub::DevicePartition only does binary partitions, so we
        // reuse arg_partition which is sort_column_pair under the hood — with
        // such a small bit range it collapses to one histogram+scan+scatter).
        auto [sorted_pid, sorted_idx] = gamax::partition::arg_partition(
            pid_col->view(), 0, log_num_partitions, nullptr, 0, stream);
        pid_col.reset();

        // Per-partition row offsets, derived from the sorted pid column.
        auto partition_ps = gamax::partition::compute_partition_offsets(
            sorted_pid->view(), 0, log_num_partitions, stream);
        sorted_pid.reset();

        // Reorder the original key column into partitioned layout. The kernel
        // discriminates groups by full-key equality (not by pid), so we need
        // the original Key values, not the pids.
        auto partitioned_key = gamax::util::device::gather(
            sorted_idx->view(), key.get_column(0), stream);

        if constexpr (kUseGFUR) {
            // GFUR: keep value columns in their original layout; the kernel
            // chases through sorted_idx (the permutation map) to fetch values.
            return compute_aggregate<Key, BucketSize, NumSlots>(
                partitioned_key->view(),
                std::move(partition_ps),
                estimated_num_groups,
                log_num_partitions,
                val,
                sorted_idx->data<gamax::size_type>(),
                agg_requests,
                stream);
        } else {
            // GFTR: pre-gather every value column into partitioned layout so
            // the kernel can stream them sequentially via a counting iterator.
            std::vector<std::unique_ptr<gamax::column>> partitioned_val_cols;
            for(gamax::size_type i = 0; i < val.num_columns(); i++) {
                partitioned_val_cols.push_back(
                    gamax::util::device::gather(sorted_idx->view(), val.get_column(i), stream));
            }
            auto partitioned_val = std::make_unique<gamax::table>(std::move(partitioned_val_cols));
            sorted_idx.reset();

            return compute_aggregate<Key, BucketSize, NumSlots>(
                partitioned_key->view(),
                std::move(partition_ps),
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
        partition_groupby_impl<bucket_size, num_slots, false>{},
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
        partition_groupby_impl<bucket_size, num_slots, true>{},
        key, val, agg_requests, hint, stream);
}
} // namespace gamax::groupby::partition