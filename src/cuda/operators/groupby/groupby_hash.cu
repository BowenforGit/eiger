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
// #include "memory/table_device_view.hpp"
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
        if(key.num_columns() > 1) {
            throw std::runtime_error("Only single-column group keys are supported");
        }

        if(gamax::size_of(key.get_column(0).type()) != 8 
            && gamax::size_of(key.get_column(0).type()) != 4) {
            throw std::runtime_error("Only 4-bytes or 8-byte group keys are supported");
        }
        
        if(key.num_columns() == 0) {
            throw std::runtime_error("Aggregation only is not supported yet");
        }

        if(agg_requests.size() == 0) {
            throw std::runtime_error("No aggregation function is specified");
        }
    }

    template<typename Key, typename Set>
    __global__ void compute_groupid_kernel(
        Set set_ref,
        const Key* keys,
        const gamax::size_type num_items,
        Key* keys_out,
        gamax::size_type* __restrict__ sparse_group_ids,
        gamax::size_type* __restrict__ group_ids_loc,
        gamax::size_type* __restrict__ num_groups)
    {
        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

        while (idx < num_items) {
            auto [iter, is_new_key] = set_ref.insert_and_find(idx);
            
            if (is_new_key) {
                sparse_group_ids[idx] = gamax::atomic::atomic_add(num_groups, static_cast<gamax::size_type>(1));
                keys_out[sparse_group_ids[idx]] = keys[idx];
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
                    res.get_column(i+1), gid);
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
                init_agg(aggs[k].agg_kind, val.get_column(k).type(), cur_agg_buffer, t);
                cur_agg_buffer = cur_agg_buffer + (num_groups)*gamax::size_of(res.get_column(k+1).type());
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
                cur_agg_buffer = cur_agg_buffer + (num_groups)*gamax::size_of(res.get_column(i+1).type());
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
                cur_agg_buffer = cur_agg_buffer + (num_groups)*gamax::size_of(res.get_column(k+1).type());
            }
            last_i = i;
            
            auto out_type = res.get_column(i+1).type();
            gamax::column temp_col(gamax::device_buffer(
                                        cur_agg_buffer, 
                                        num_groups*gamax::size_of(out_type)), 
                                    num_groups, 
                                    out_type);
            gamax::groupby::details::aggregate(
                aggs[i].agg_kind,
                temp_col.view(), j,
                res.get_column(i+1), j);
        }
    }

    template <typename T>
    struct my_equal {
        my_equal(const T* data) : _data{data} {}
        __device__ constexpr bool operator()(gamax::size_type lhs, gamax::size_type rhs) const
        {
            // Check all 4 elements of a tuple to determine if two tuples are identical
            return _data[lhs] == _data[rhs];
        }
        const T* _data;
    };

    template <typename T>
    struct my_hasher {
        my_hasher(const T* data) : _data{data} {}
        __device__ auto operator()(gamax::size_type index) const
        {
            return cuco::default_hash_function<T>{}(_data[index]);
        }
        const T* _data;
    };

    struct hash_groupby_fast_impl {
        template<typename Key>
        std::unique_ptr<gamax::table> operator()(
            gamax::table_view key,
            gamax::table_view val,
            std::vector<gamax::groupby::aggregation_request>& agg_requests,
            gamax::groupby::groupby_hint& hint,
            cudaStream_t stream = cudaStreamDefault)
        {
            auto input_num_rows = key.num_rows();
            gamax::size_type estimated_num_groups = std::min(input_num_rows, hint.groups_upper_bound);

            auto d_agg_req_buf = std::make_unique<gamax::device_buffer>(sizeof(gamax::groupby::aggregation_request) * agg_requests.size(), false);
            cudaMemcpy(d_agg_req_buf->data(), agg_requests.data(), d_agg_req_buf->size(), cudaMemcpyHostToDevice);
            
            std::vector<std::unique_ptr<gamax::column>> final_columns;
            final_columns.push_back(
                gamax::make_column(
                    key.get_column(0).type(), 
                    estimated_num_groups, 
                    false));
            for(const auto& agg : agg_requests) {
                final_columns.push_back(
                    gamax::make_column(
                        gamax::groupby::agg_output_data_type(agg.agg_kind, val.get_column(agg.column_index).type()), 
                        estimated_num_groups));
            }
            auto res = std::make_unique<gamax::table>(std::move(final_columns));
            
            constexpr double load_factor = 0.25;
            std::size_t map_capacity = std::ceil(std::max(static_cast<gamax::size_type>(4096), estimated_num_groups) / load_factor);
            gamax::size_type constexpr empty_key_sentinel     = gamax::sentinel<gamax::size_type>(); // FIXME: handle negative values and unsigned values correctly       
            auto cuco_allocator = *get_thrust_allocator<cuco::pair<Key, gamax::size_type>>(stream);   

            auto key_ptr = key.get_column(0).data<Key>();

            auto set = cuco::static_set{estimated_num_groups,
                                    load_factor,
                                    cuco::empty_key{empty_key_sentinel},
                                    my_equal<Key>{key_ptr},
                                    cuco::linear_probing<1, my_hasher<Key>>{my_hasher{key_ptr}},
#ifdef __NVCC__ // only supported for CUCO
                                    cuco::thread_scope_device,
                                    cuco::storage<1>{},
#endif
                                    cuco_allocator, 
                                    stream
                                    };
            auto set_ref = set.ref(cuco::insert_and_find);

            auto sparse_group_ids = gamax::make_idx_column(input_num_rows, false);
            auto groupid_loc = gamax::make_idx_column(input_num_rows, false);
            auto d_num_groups = gamax::make_idx_column(1, true);

            compute_groupid_kernel<<<num_tb(input_num_rows, 256), 256, 0, stream>>>(
                set_ref, 
                key.get_column(0).data<Key>(), 
                input_num_rows, 
                res->get_column(0).data<Key>(),
                sparse_group_ids->data<gamax::size_type>(), 
                groupid_loc->data<gamax::size_type>(), 
                d_num_groups->data<gamax::size_type>());

            // map.find(key.get_column(0).data<Key>(), key.get_column(0).data<Key>() + input_num_rows, groupid_loc->data<gamax::size_type>(), cuco::cuda_stream_ref{stream});

            gamax::size_type num_groups;
            cudaMemcpyAsync(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // auto val_view = table_device_view::create(val);
            // auto res_view = table_device_view::create(*res);
            auto val_view = val.device_view(stream);
            auto res_view = res->mutable_view().device_view(stream); // mutable

            initialize_aggregation<<<num_tb(res->num_rows(), 1024), 1024, 0, stream>>>(*val_view, *res_view, static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()));
            
            std::size_t agg_bytes = 0;
            for(gamax::size_type i = 1; i < res->num_columns(); i++) {
                agg_bytes += gamax::size_of(res->get_column(i).type());
            }
            
            constexpr std::size_t max_shared_mem_size = MAX_SMEM_BYTES;
            /* When the number of groups becomes larger, 
            it is no longer efficient to aggregate in SMEM 
            even though it goes not go beyond SMEM's capacity.
            Experimentally, we found 2048 to be the threshold */
            const auto smem_groups = std::min(max_shared_mem_size / agg_bytes, 16384UL);

            if(num_groups > smem_groups) {
                 compute_aggregation<<<num_tb(input_num_rows, 1024), 1024, 0, stream>>>(
                    *val_view, 
                    *res_view, 
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
                // std::cout << "sm_bytes: " << sm_bytes << std::endl;
                // agg_fn<<<num_tb(input_num_rows, 768), 768, sm_bytes, stream>>>(
                int maxActiveBlocks;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, agg_fn, 1024, 0);
                agg_fn<<<SM_COUNT*maxActiveBlocks, 1024, sm_bytes, stream>>>(
                    *val_view, 
                    *res_view, 
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
    };
} // unnamed namespace

// currently only sum and max are supported
std::unique_ptr<gamax::table> groupby(
    gamax::table_view key, 
    gamax::table_view val, 
    std::vector<gamax::groupby::aggregation_request>& agg_requests,
    gamax::groupby::groupby_hint& hint,
    cudaStream_t stream)
{
    check_solvable(key, val, agg_requests);
    
    return type_dispatcher(
        key.get_column(0).type(),
        hash_groupby_fast_impl{},
        key, val, agg_requests, hint, stream);
}
} // namespace gamax::groupby::hash
