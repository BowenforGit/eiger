#pragma once

#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <tuple>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 

#include "load_balancer.cuh"
#include "types/tuple.cuh"
#include "utils/utils.cuh"
#include "utils/cub_wrapper.cuh"
#include "operators/partition/partition.cuh"
#include "containers/agg_hashmap.cuh"

namespace gamax {
namespace groupby {
namespace partition {
    template<typename F, 
             typename KeyT, 
             typename ValT, 
             typename NumItemsT, 
             typename IdxT, 
             typename IntermediateT, 
             typename ResultT, 
             int VT=4>
    __global__ void aggregate_kernel_block_monoid(KeyT* keys,
                                                ValT* values, 
                                                NumItemsT num_units,
                                                IdxT* units_start, 
                                                IdxT* units_end, 
                                                IdxT* units_group, 
                                                const int log_parts,
                                                NumItemsT* num_groups,
                                                IntermediateT* intermediate_results, 
                                                KeyT* grouped_keys,
                                                ResultT* final_results) {
        static_assert(F::is_monoid && std::is_same_v<ResultT, typename F::OutputType>);
        
        extern __shared__ int16_t temp[];

        int max_partition_size = 4096;
        int hash_table_capacity = max_partition_size+512;
        auto hash_fn = [&](const KeyT& k) -> int {
            return (k >> log_parts) & (hash_table_capacity - 1);
        };
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
                        if(e == 1) printf("insert failed\n");
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
             typename ValT, 
             typename NumItemsT, 
             typename IdxT, 
             typename IntermediateT, 
             typename ResultT>
    void aggregate(KeyT* keys,
                    ValT* values, 
                    NumItemsT num_units,
                    IdxT* units_start, 
                    IdxT* units_end, 
                    IdxT* units_group, 
                    IntermediateT* intermediate_results, 
                    const int log_parts,
                    NumItemsT* num_groups,
                    KeyT* grouped_keys,
                    ResultT* final_results) {
        constexpr int NT = 256;

        NumItemsT* d_num_groups = nullptr;
        allocate_mem(&d_num_groups, true, sizeof(NumItemsT));
        size_t sm_bytes = (4096*4*2*2); // TODO: Fix this
        auto agg_fn = aggregate_kernel_block_monoid<F, KeyT, ValT, NumItemsT, IdxT, IntermediateT, ResultT, 1>;
        cudaFuncSetAttribute(agg_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes);
        agg_fn<<<num_units, NT, sm_bytes>>>(keys, values, 
                                                            num_units, units_start, 
                                                            units_end, units_group, 
                                                            log_parts, d_num_groups,
                                                            intermediate_results, 
                                                            grouped_keys, final_results);
        cudaMemcpy(num_groups, d_num_groups, sizeof(NumItemsT), cudaMemcpyDeviceToHost);
        std::cout << "num_groups: " << *num_groups << std::endl;
        release_mem(d_num_groups);
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

        int log_num_partitions = 16;
        int num_partitions = 1 << log_num_partitions;

        gamax::partition::Partition p(keys, vals, num_items, log_num_partitions);
        auto partitioned_keys = p.get_partitioned_keys();
        auto partitioned_vals = p.get_partitioned_values();
        auto partition_ps = p.get_partition_offsets();

        constexpr IdxT threshold = std::numeric_limits<IdxT>::max() / 2; // TODO: be careful of overflow
        IdxT* units_start = nullptr, *units_end = nullptr, *units_group = nullptr;
        int max_work_units = num_partitions + (num_items+threshold-1)/threshold;
        allocate_mem(&units_start, false, max_work_units*sizeof(IdxT));
        allocate_mem(&units_end, false, max_work_units*sizeof(IdxT));
        allocate_mem(&units_group, false, max_work_units*sizeof(IdxT));
        int num_units = 0;
        generate_workunits(partition_ps, num_partitions, threshold, &num_units, units_start, units_end, units_group);

        auto unique_keys = tout.template get_typed_ptr<0>();
        int num_groups = 0;

        aggregate<AggFn>(partitioned_keys, partitioned_vals, 
                        num_units, 
                        units_start, units_end, units_group,
                        COL(tout, 1), 
                        log_num_partitions, &num_groups,
                        COL(tout, 0), COL(tout, 1));
        tout.num_items = num_groups;

        cudaStreamSynchronize(0);

        release_mem(units_start);
        release_mem(units_end);
        release_mem(units_group);

        // std::cout << "grouped_agg done" << std::endl;
        // debug_cuda_mem_usage();
    }
} // namespace partition
} // namespace groupby
} // namespace gamax