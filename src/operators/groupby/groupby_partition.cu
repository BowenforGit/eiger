#include "operators/groupby/groupby_partition.hpp"

#include <memory>
#include <cuda.h>
#include <cub/cub.cuh> 

#include "utils/utils.cuh"
#include "utils/cub_wrapper.cuh"
#include "containers/agg_hashmap.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/partition/partition.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/load_balancer.cuh"
#include "operators/groupby/agg_func.cuh"
#include "types/data_type.hpp"

namespace gamax::groupby::partition {
namespace {
template<typename F, 
            typename KeyT, 
            typename ValT, 
            typename NumItemsT, 
            typename IdxT, 
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
            typename ValT, 
            typename NumItemsT, 
            typename IdxT, 
            typename ResultT>
void aggregate(KeyT* keys,
                ValT* values, 
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
    auto agg_fn = aggregate_kernel_block_monoid<F, KeyT, ValT, NumItemsT, IdxT, ResultT, 1>;
    cudaFuncSetAttribute(agg_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes);
    agg_fn<<<num_units, NT, sm_bytes>>>(keys, values, 
                                                        num_units, units_start, 
                                                        units_end, units_group, 
                                                        log_parts, d_num_groups,
                                                        grouped_keys, final_results);
    cudaMemcpy(num_groups, d_num_groups, sizeof(NumItemsT), cudaMemcpyDeviceToHost);
    release_mem(d_num_groups);
}
}

std::unique_ptr<gamax::table> groupby(
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
} // namespace gamax::groupby::partition