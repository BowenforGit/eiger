#pragma once

#define CUB_STDERR

#include <iostream>
#include <tuple>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 

#include "types/tuple.cuh"
#include "utils/utils.cuh"
#include "../join/partition_util.cuh"

namespace gamax {
    namespace partition {
        template <typename KeyT, typename ValueT, typename NumItemsT>
        class Partition {
        public:
            Partition(KeyT* keys, ValueT* values, NumItemsT num_items, int log_num_partitions, cudaStream_t stream = 0) 
                : keys(keys), values(values), num_items(num_items), log_num_partitions(log_num_partitions), stream(stream) {
                num_partitions = 1 << log_num_partitions;

                allocate_mem(&partition_offsets, false, ((1 << log_num_partitions) + 1)*sizeof(NumItemsT));
                allocate_mem(&partitioned_keys, false, num_items*sizeof(KeyT));
                allocate_mem(&partitioned_values, false, num_items*sizeof(ValueT));

                partition();
            }

            Partition(KeyT* keys, ValueT* values, NumItemsT num_items, int log_num_partitions, NumItemsT* partition_offsets, cudaStream_t stream = 0) 
                : keys(keys)
                , values(values)
                , num_items(num_items)
                , log_num_partitions(log_num_partitions)
                , partition_offsets(partition_offsets)
                , stream(stream) {
                num_partitions = 1 << log_num_partitions;
                calculate_offsets = false;
                
                allocate_mem(&partitioned_keys, false, num_items*sizeof(KeyT));
                allocate_mem(&partitioned_values, false, num_items*sizeof(ValueT));

                partition();
            }

            ~Partition() {
                release_mem(partitioned_keys);
                release_mem(partitioned_values);
                if(calculate_offsets) release_mem(partition_offsets);
            }

            KeyT* get_partitioned_keys() {
                return partitioned_keys;
            }

            ValueT* get_partitioned_values() {
                return partitioned_values;
            }

            NumItemsT* get_partition_offsets() {
                return partition_offsets;
            }
        
        private:
            void partition() {
                SinglePassPartition<KeyT, ValueT, NumItemsT> agent(keys, values, 
                                          partitioned_keys, partitioned_values, 
                                          (calculate_offsets ? partition_offsets : nullptr), 
                                          (long)num_items, 
                                          0, log_num_partitions);
                agent.process();
            }
        
        private:
            KeyT* keys {nullptr};
            ValueT* values {nullptr};
            KeyT* partitioned_keys {nullptr};
            ValueT* partitioned_values {nullptr};
            NumItemsT* partition_offsets {nullptr};
            bool calculate_offsets {true};
            NumItemsT num_items;
            int log_num_partitions;
            int num_partitions;
            cudaStream_t stream;
        };
    } // namespace partition
} // namespace gamax