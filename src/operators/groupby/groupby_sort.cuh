#pragma once

#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <tuple>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 

#include "types/tuple.cuh"
#include "utils/utils.cuh"
#include "utils/cub_wrapper.cuh"
#include "../sort/sort.cuh"

/* Sort-based implementation 

1. Sort the group keys
2. Assign a unique group id to each group
3. Sort the payload columns based on the group keys

Questions:
1. How to sort multiple group keys efficiently?
2. How to sort the payload columns only once to avoid excessive data movement?
*/

namespace gamax {
namespace groupby {
namespace sort {
    template<typename EncoderT, typename Tuple, typename EncodedKeyT>
    void encode_groupkeys(Tuple t, EncodedKeyT* encoded_keys) {
        // encode the group keys
        // for each group key, encode it and store the encoded value in the encoded_keys array
        EncoderT encoder(t);
        encoded_keys = encoder.encode();
    }
    
    template<typename T, typename ValT, typename NumItemsT, typename IdxT>
    void group_keys(T* keys_in, 
                    T* unique_keys_out, 
                    ValT* vals_in,
                    ValT* vals_out,
                    NumItemsT num_items,
                    NumItemsT* num_groups, 
                    IdxT* membership_ps) {
        // IdxT* tids = nullptr;
        // allocate_mem(&tids, true, num_items*sizeof(IdxT));
        // fill_sequence<<<num_tb(num_items, 1024), 1024>>>(tids, 0, static_cast<size_t>(num_items));
        
        T* keys_out = nullptr;
        allocate_mem(&keys_out, false, num_items*sizeof(T));
        gamax::sort::radix_sort_kv(keys_in, keys_out, vals_in, vals_out, num_items);

        IdxT* membership_cnt = nullptr;
        allocate_mem(&membership_cnt, true, num_items*sizeof(IdxT));
        NumItemsT* num_runs = nullptr;
        allocate_mem(&num_runs, true, sizeof(NumItemsT));
        CALL_CUB_FUNC(cub::DeviceRunLengthEncode::Encode, keys_out, unique_keys_out, membership_cnt, num_runs, num_items)

        CHECK_CUDA_ERROR(cudaMemcpy(num_groups, num_runs, sizeof(NumItemsT), cudaMemcpyDeviceToHost));
        CALL_CUB_FUNC(cub::DeviceScan::InclusiveSum, membership_cnt, membership_ps, *num_groups)

        // release_mem(tids);
        release_mem(keys_out);
        release_mem(membership_cnt);
        release_mem(num_runs);
    }

    template<typename NumItemsT, typename IdxT>
    __global__ void generate_workunits_kernel(const IdxT* membership_ps, 
                            const NumItemsT num_groups,
                            const IdxT threshold,
                            NumItemsT* num_units,
                            IdxT* units_start, 
                            IdxT* units_end, 
                            IdxT* units_group) {
        for(int i = get_cuda_tid(); i < num_groups; i += nthreads()) {
            IdxT start = (i == 0) ? 0 : membership_ps[i-1];
            IdxT end = membership_ps[i];
            IdxT num_items = end - start;
            IdxT num_work_units = (num_items+threshold-1)/threshold;
            for(int j = 0; j < num_work_units; j++) {
                IdxT unit_start = j*threshold;
                IdxT unit_end = min((j+1)*threshold, num_items);
                int idx = atomicAdd(num_units, 1);
                units_start[idx] = start+unit_start;
                units_end[idx] = start+unit_end;
                units_group[idx] = i;
            }
        }
    }

    template<typename NumItemsT, typename IdxT>
    void generate_workunits(const IdxT* membership_ps, 
                            const NumItemsT num_groups,
                            const IdxT threshold,
                            NumItemsT* num_units,
                            IdxT* units_start, 
                            IdxT* units_end, 
                            IdxT* units_group) {
        NumItemsT* d_num_units = nullptr;
        allocate_mem(&d_num_units, true, sizeof(NumItemsT));
        generate_workunits_kernel<<<num_tb(num_groups, 1024), 1024>>>(membership_ps, num_groups, threshold, d_num_units, units_start, units_end, units_group);
        CHECK_CUDA_ERROR(cudaMemcpy(num_units, d_num_units, sizeof(NumItemsT), cudaMemcpyDeviceToHost));
        release_mem(d_num_units);
    }

    template<typename F, int NT, typename T, typename NumItemsT, typename IdxT, typename IntermediateT, typename ResultT>
    __global__ void aggregate_kernel_block_monoid(T* values, 
                    NumItemsT num_items, 
                    NumItemsT num_units,
                    IdxT* units_start, 
                    IdxT* units_end, 
                    IdxT* units_group, 
                    const NumItemsT num_groups, 
                    IntermediateT* intermediate_results, 
                    ResultT* final_results) {
        static_assert(F::is_monoid);
        
        F f;
        ResultT thread_local_agg = F::identity;
        
        #pragma unroll
        for(int i = blockIdx.x; i < num_units; i += gridDim.x) {
            IdxT group_id = units_group[i];
            
            #pragma unroll
            for(int t = units_start[i]+threadIdx.x; t < units_end[i]; t += blockDim.x) {
                // f.update(group_id, final_results, intermediate_results, values[t]);
                thread_local_agg = f.thread_local_update(thread_local_agg, values[t]);
            }
            __syncthreads();
            
            // block reduction
            typedef cub::BlockReduce<decltype(thread_local_agg), NT> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            ResultT block_agg = BlockReduce(temp_storage).Reduce(thread_local_agg, f);
            if(threadIdx.x == 0) { // only thread-0 has the correct aggregate
                f.global_update(group_id, final_results, intermediate_results, block_agg);
            }

            thread_local_agg = F::identity;
        }
    }

    template<typename F, typename T, typename NumItemsT, typename IdxT, typename IntermediateT, typename ResultT>
    void aggregate(T* values, 
                    NumItemsT num_items, 
                    NumItemsT num_units,
                    IdxT* units_start, 
                    IdxT* units_end, 
                    IdxT* units_group, 
                    const NumItemsT num_groups,
                    IntermediateT* intermediate_results, 
                    ResultT* final_results) {
        // each package: (start, end, group_id)
        // assign one thread block per work package
        constexpr int NT = 256;
        aggregate_kernel_block_monoid<F, NT><<<num_units, NT>>>(values, num_items, num_units, units_start, units_end, units_group, num_groups, intermediate_results, final_results);
    }

    template<typename DecoderT, typename Tuple, typename EncodedKeyT>
    void decode_groupkeys(Tuple t, EncodedKeyT* encoded_keys) {
        DecoderT decoder(t, encoded_keys);
        decoder.decode();
    }

    // the most generic version of the groupby function
    template<typename TupleKeys, 
             typename TupleVals, 
             typename TupleOut,
             typename EncoderT,
             typename DecoderT,
             typename... Fs>
    void grouped_agg(TupleKeys tkeys, TupleVals tvals, TupleOut& tout) {

    }

    // a simplistic version that only supports single-column group keys and single-column payload
    template<typename AggFn,
             typename TupleKeys, 
             typename TupleVals, 
             typename TupleOut>
    void grouped_agg(TupleKeys tkeys, TupleVals tvals, TupleOut& tout) {
        static_assert(TupleKeys::num_cols == 1 && TupleVals::num_cols == 1);
        static_assert(TupleKeys::num_cols+TupleVals::num_cols == TupleOut::num_cols);
        assert(tkeys.num_items == tvals.num_items);

        using key_t = std::tuple_element_t<0, typename TupleKeys::value_type>;
        using val_t = std::tuple_element_t<0, typename TupleVals::value_type>;
        using IdxT = decltype(tkeys.num_items);

        auto keys = tkeys.template get_typed_ptr<0>();
        auto vals = tvals.template get_typed_ptr<0>();

        auto num_items = tkeys.num_items;

        auto unique_keys = tout.template get_typed_ptr<0>();
        IdxT* membership_ps = nullptr;
        // IdxT* shuffled_tids = nullptr;
        allocate_mem(&membership_ps, false, num_items*sizeof(IdxT));
        // allocate_mem(&shuffled_tids, false, num_items*sizeof(IdxT));
        int num_groups = 0;
        
        val_t* sorted_vals = nullptr;
        allocate_mem(&sorted_vals, false, num_items*sizeof(val_t));
        group_keys(keys, unique_keys, COL(tvals, 0), sorted_vals, num_items, &num_groups, membership_ps);
        // print_gpu_arr(unique_keys, 16);
        // print_gpu_arr(membership_ps, 16);
        // std::cout << "num_groups: " << num_groups << std::endl;

        // if constexpr (sizeof(key_t) > 2*sizeof(IdxT)) {
        //     IdxT* tids = nullptr, *shuffled_tids_sorted = nullptr, *tids_sorted = nullptr;
        //     allocate_mem(&tids, false, num_items*sizeof(IdxT));
        //     allocate_mem(&shuffled_tids_sorted, false, num_items*sizeof(IdxT));
        //     allocate_mem(&tids_sorted, false, num_items*sizeof(IdxT));
        //     fill_sequence<<<num_tb(num_items, 1024), 1024>>>(tids, 0, static_cast<size_t>(num_items));

        //     gamax::sort::radix_sort_kv(shuffled_tids, shuffled_tids_sorted, tids, tids_sorted, num_items);
        //     release_mem(shuffled_tids_sorted);

        //     gamax::sort::radix_sort_kv(tids_sorted, tids, vals, sorted_vals, num_items);
        //     release_mem(tids);
        //     release_mem(tids_sorted);
        // } else {
        //     key_t* sorted_keys = nullptr;
        //     allocate_mem(&sorted_keys, false, num_items*sizeof(key_t));
        //     gamax::sort::radix_sort_kv(keys, sorted_keys, vals, sorted_vals, num_items);
        //     release_mem(sorted_keys);
        // }

        constexpr IdxT threshold = 4096;
        IdxT* units_start = nullptr, *units_end = nullptr, *units_group = nullptr;
        int max_work_units = num_groups + (num_items+threshold-1)/threshold;
        allocate_mem(&units_start, false, max_work_units*sizeof(IdxT));
        allocate_mem(&units_end, false, max_work_units*sizeof(IdxT));
        allocate_mem(&units_group, false, max_work_units*sizeof(IdxT));
        int num_units = 0;
        generate_workunits(membership_ps, num_groups, threshold, &num_units, units_start, units_end, units_group);
        // print_gpu_arr(units_start, 16);
        // print_gpu_arr(units_end, 16);
        // print_gpu_arr(units_group, 16);

        aggregate<AggFn>(sorted_vals, num_items, num_units, units_start, units_end, units_group, num_groups, tout.template get_typed_ptr<1>(), tout.template get_typed_ptr<1>());
        tout.num_items = num_groups;

        cudaStreamSynchronize(0);

        release_mem(membership_ps);
        // release_mem(shuffled_tids);
        release_mem(sorted_vals);
        release_mem(units_start);
        release_mem(units_end);
        release_mem(units_group);

        // std::cout << "grouped_agg done" << std::endl;
        // debug_cuda_mem_usage();
    }
} // namespace sort
} // namespace groupby
} // namespace gamax