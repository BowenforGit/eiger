#include "operators/groupby/groupby_partition.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
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
constexpr std::size_t num_slots = 8209;

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

// pid is bounded by num_partitions ≤ 2^16, so int32 (4B) is sufficient and
// halves both the column footprint (~2.4 GiB saved at SF=100 lineitem widths)
// and the per-pair CUB radix sort key width vs. size_type (int64).
template<typename Key>
__global__ void compute_partition_ids_kernel(
    const Key* __restrict__ keys,
    std::int32_t* __restrict__ pids,
    gamax::size_type n,
    std::uint32_t mask)
{
    for(gamax::size_type t = get_cuda_tid(); t < n; t += nthreads()) {
        pids[t] = static_cast<std::int32_t>(bit_mix(keys[t]) & mask);
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

// Env-var gated diagnostic. When GAMAX_PARTITION_OVERFLOW_DIAG is set,
// copies the partition-major key column + inclusive prefix-sum offsets
// to host, then for each partition runs std::sort + std::unique to
// count distinct keys. Reports the top-K, the max, and a coarse
// distribution. Used to root-cause "per-partition slot table overflow"
// throws: it confirms which partition saturates and by how much.
//
// Cost at SF=100: ~5 GiB host copy + 65k partitions × ~9k-row sort ≈
// a few seconds. Off by default — set the env var to opt in.
template<typename Key>
void diagnose_partition_distinct_counts(
    gamax::column_view partitioned_key,
    const gamax::size_type* partition_ps_inclusive_device,
    gamax::size_type num_partitions,
    std::size_t num_slots_per_partition,
    cudaStream_t stream)
{
    const auto N = partitioned_key.size();
    std::cerr << "[partition diag] copying partitioned_key (" << N
              << " rows × " << sizeof(Key) << " B = "
              << (double(N) * sizeof(Key) / (1024.0 * 1024.0 * 1024.0))
              << " GiB) and offsets to host..." << std::endl;

    // std::vector<bool> has no .data() member, but the partition_groupby
    // template instantiates for every integral Key — use a raw buffer.
    auto h_keys = std::make_unique<Key[]>(static_cast<std::size_t>(N));
    cudaMemcpyAsync(h_keys.get(), partitioned_key.template data<Key>(),
                    static_cast<std::size_t>(N) * sizeof(Key),
                    cudaMemcpyDeviceToHost, stream);

    // partition_ps is an INCLUSIVE prefix sum of length num_partitions.
    //   partition_ps[i] = total rows in partitions [0..i].
    // Partition i's row range: [(i==0)?0:partition_ps[i-1], partition_ps[i]).
    std::vector<gamax::size_type> h_ps(num_partitions);
    cudaMemcpyAsync(h_ps.data(), partition_ps_inclusive_device,
                    num_partitions * sizeof(gamax::size_type),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<gamax::size_type> h_counts(num_partitions, 0);
    std::vector<gamax::size_type> h_sizes(num_partitions, 0);
    gamax::size_type max_distinct = 0;
    gamax::size_type max_partition = 0;
    gamax::size_type overflows = 0;
    gamax::size_type empties = 0;
    gamax::size_type total_distinct = 0;

    auto slice = std::make_unique<Key[]>(static_cast<std::size_t>(N));
    for (gamax::size_type p = 0; p < num_partitions; ++p) {
        gamax::size_type start = (p == 0) ? gamax::size_type{0} : h_ps[p - 1];
        gamax::size_type end = h_ps[p];
        gamax::size_type sz = end - start;
        h_sizes[p] = sz;
        if (sz == 0) { ++empties; continue; }

        std::copy(h_keys.get() + start, h_keys.get() + end, slice.get());
        std::sort(slice.get(), slice.get() + sz);
        auto new_end = std::unique(slice.get(), slice.get() + sz);
        gamax::size_type distinct = static_cast<gamax::size_type>(new_end - slice.get());

        h_counts[p] = distinct;
        total_distinct += distinct;
        if (distinct >= static_cast<gamax::size_type>(num_slots_per_partition)) ++overflows;
        if (distinct > max_distinct) {
            max_distinct = distinct;
            max_partition = p;
        }
    }

    std::cerr << "[partition diag] N=" << N
              << " num_partitions=" << num_partitions
              << " NumSlots(per-partition cap)=" << num_slots_per_partition
              << "\n[partition diag] total_distinct=" << total_distinct
              << " empty_partitions=" << empties
              << " overflowing_partitions(distinct>=NumSlots)=" << overflows
              << "\n[partition diag] max_distinct=" << max_distinct
              << " in partition " << max_partition
              << " (rows=" << h_sizes[max_partition] << ")"
              << std::endl;

    // Top-K partitions by distinct count
    std::vector<gamax::size_type> idx(num_partitions);
    for (gamax::size_type p = 0; p < num_partitions; ++p) idx[p] = p;
    const int K = static_cast<int>(std::min<gamax::size_type>(20, num_partitions));
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](gamax::size_type a, gamax::size_type b) {
                          return h_counts[a] > h_counts[b];
                      });
    std::cerr << "[partition diag] Top " << K
              << " partitions by distinct key count:\n";
    for (int i = 0; i < K; ++i) {
        auto p = idx[i];
        std::cerr << "  partition[" << p << "]:"
                  << " distinct=" << h_counts[p]
                  << " rows=" << h_sizes[p]
                  << " dup_ratio=" << (h_sizes[p] > 0
                                        ? double(h_sizes[p]) / double(h_counts[p])
                                        : 0.0)
                  << "\n";
    }

    // Distribution histogram
    const gamax::size_type nslots = static_cast<gamax::size_type>(num_slots_per_partition);
    const std::vector<gamax::size_type> bounds = {
        0, 1, 1000, 4000, 7000, nslots, nslots + nslots / 4, nslots * 2,
        nslots * 5, std::numeric_limits<gamax::size_type>::max()};
    std::cerr << "[partition diag] distribution buckets [lo, hi):" << std::endl;
    for (std::size_t b = 0; b + 1 < bounds.size(); ++b) {
        gamax::size_type cnt = 0;
        for (gamax::size_type p = 0; p < num_partitions; ++p) {
            if (h_counts[p] >= bounds[b] && h_counts[p] < bounds[b + 1]) ++cnt;
        }
        std::cerr << "  [" << bounds[b] << ", ";
        if (bounds[b + 1] == std::numeric_limits<gamax::size_type>::max()) std::cerr << "inf";
        else std::cerr << bounds[b + 1];
        std::cerr << "): " << cnt << " partitions" << std::endl;
    }
    std::cerr.flush();
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

    // Opt-in diagnostic: when GAMAX_PARTITION_OVERFLOW_DIAG is set,
    // count distinct keys per partition on the host and report the
    // worst offenders. Useful to root-cause "per-partition slot table
    // overflow" throws.
    if (std::getenv("GAMAX_PARTITION_OVERFLOW_DIAG")) {
        diagnose_partition_distinct_counts<Key>(
            partitioned_key,
            partition_ps->data<gamax::size_type>(),
            num_partitions,
            NumSlots,
            stream);
    }

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
        const std::uint32_t mask = (std::uint32_t{1} << log_num_partitions) - 1;

        // Compute per-row partition_id = bit_mix(key) & mask. Bit-mixing the key
        // before extracting low radix bits prevents skew when the key has low
        // entropy in its low bits (e.g. TPC-H orderkeys are multiples of 4, so a
        // plain low-bit radix would empty 3/4 of the partitions and overload the
        // remaining 1/4 past the per-block slot table capacity). pid fits in
        // int32 since num_partitions ≤ 2^16, so the column is 4B/row.
        auto pid_col = gamax::make_column<std::int32_t>(input_num_rows, false, stream);
        {
            const auto threads = 1024;
            const auto blocks = num_tb(input_num_rows, threads);
            compute_partition_ids_kernel<Key><<<blocks, threads, 0, stream>>>(
                key.get_column(0).data<Key>(),
                pid_col->data<std::int32_t>(),
                input_num_rows,
                mask);
        }

        if constexpr (kUseGFUR) {
            // GFUR: one multi-way partition over pid produces (sorted_pid,
            // sorted_idx). The kernel chases through sorted_idx (the
            // permutation map) to fetch each value at aggregation time —
            // GFUR's defining trade-off is "pay random gather inside the
            // kernel; skip the per-payload partition pass."
            auto [sorted_pid, sorted_idx] = gamax::partition::arg_partition(
                pid_col->view(), 0, log_num_partitions, nullptr, 0, stream);
            pid_col.reset();

            auto partition_ps = gamax::partition::compute_partition_offsets(
                sorted_pid->view(), 0, log_num_partitions, stream);
            sorted_pid.reset();

            // The kernel discriminates groups by full-key equality, so it
            // needs the original Key values in partitioned layout. A single
            // gather is sufficient and cheaper than re-sorting just for the
            // key column.
            auto partitioned_key = gamax::util::device::gather(
                sorted_idx->view(), key.get_column(0), stream);

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
            // GFTR: each payload column is partitioned alongside pid via its
            // own `partition_column_pair(pid, X)` — one CUB radix sort per
            // payload that carries (pid, X) into partitioned layout. The
            // kernel can then stream payloads sequentially with a counting
            // iterator; no random gather inside the agg loop. This mirrors
            // the original (pre-bit-mix) GFTR loop, with `pid` substituted
            // for the raw key as the sort discriminator. The per-payload
            // sort is the whole point of GFTR — replacing it with a single
            // arg_partition + N gathers would defeat the algo.
            std::unique_ptr<gamax::column> sorted_pid;
            std::unique_ptr<gamax::column> partitioned_key;
            std::tie(sorted_pid, partitioned_key) =
                gamax::partition::partition_column_pair(
                    pid_col->view(), key.get_column(0),
                    0, log_num_partitions, nullptr, 0, stream);

            std::vector<std::unique_ptr<gamax::column>> partitioned_val_cols;
            partitioned_val_cols.reserve(val.num_columns());
            for (gamax::size_type i = 0; i < val.num_columns(); i++) {
                std::unique_ptr<gamax::column> pval_temp;
                std::tie(sorted_pid, pval_temp) =
                    gamax::partition::partition_column_pair(
                        pid_col->view(), val.get_column(i),
                        0, log_num_partitions, nullptr, 0, stream);
                partitioned_val_cols.push_back(std::move(pval_temp));
            }
            pid_col.reset();

            auto partition_ps = gamax::partition::compute_partition_offsets(
                sorted_pid->view(), 0, log_num_partitions, stream);
            sorted_pid.reset();

            auto partitioned_val =
                std::make_unique<gamax::table>(std::move(partitioned_val_cols));

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