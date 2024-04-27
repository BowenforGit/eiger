#include "operators/groupby/groupby_hash.hpp"

#include <cmath>
#include <cuco/static_map.cuh>
#include <cuda.h>
#include <cuda/std/atomic>
#include <memory>
#include <stdexcept>
#include <thrust/functional.h>
#include <type_traits>

#include "utils/utils.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/table_device_view.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "types/data_type.hpp"
#include "utils/transform.cuh"

namespace gamax::groupby::hash {
namespace {
    void check_solvable(gamax::table& key, gamax::table& val, std::vector<gamax::groupby::aggregation_request>& agg_requests) {
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

    template<gamax::groupby::aggregation_id agg_kind>
    struct dispatch_val_type_helper {
        template<typename T>
        __device__ __forceinline__ void operator()(
            gamax::column& src,
            gamax::size_type src_idx,
            gamax::column& dst,
            gamax::size_type dst_idx)
        {
            using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;
            F{}(src, src_idx, dst, dst_idx);
        }

        __device__ __forceinline__ void operator()(
            gamax::column& src,
            gamax::size_type src_idx,
            gamax::column& dst,
            gamax::size_type dst_idx)
        {

            type_dispatcher(
                src.type(),
                dispatch_val_type_helper<agg_kind>{},
                src, 
                src_idx, 
                dst, 
                dst_idx);
        }
    };

    template<typename... Ts>
    __device__ __forceinline__ void dispatch_type_aggregate(gamax::groupby::aggregation_id agg_kind, Ts&&... args)
    {
        switch(agg_kind) {
            case gamax::groupby::aggregation_id::SUM:
                dispatch_val_type_helper<gamax::groupby::aggregation_id::SUM>{}(std::forward<Ts>(args)...);
                break;
            case gamax::groupby::aggregation_id::MAX:
                dispatch_val_type_helper<gamax::groupby::aggregation_id::MAX>{}(std::forward<Ts>(args)...);
                break;
            default:
                GAMAX_UNREACHABLE("Unsupported aggregation function");
        }
    }

    __device__ __forceinline__ void aggregate(
        gamax::groupby::aggregation_id agg_kind,
        gamax::column& src,
        gamax::size_type src_idx,
        gamax::column& dst,
        gamax::size_type dst_idx)
    {
        dispatch_type_aggregate(
            agg_kind,
            src, 
            src_idx, 
            dst, 
            dst_idx);
    }

    template<typename Key, typename Map>
    __global__ void hash_groupby_kernel(
        Map map_ref,
        gamax::table_device_view key,
        gamax::table_device_view val,
        gamax::table_device_view res,
        gamax::groupby::aggregation_request* aggs,
        const gamax::size_type num_aggs,
        gamax::size_type* num_groups)
    {
        auto num_items = key.num_rows();

        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

        auto key_data = key[0].data<Key>();

        gamax::size_type unintialized_flag = -2;
        while (idx < num_items) {
            auto [slot, is_new_key] = map_ref.insert_and_find(cuco::pair{key_data[idx], unintialized_flag});
            auto ref = cuda::atomic_ref<gamax::size_type, cuda::thread_scope_device>{slot->second};
            
            gamax::size_type t_out_pos{0};
            if (is_new_key) {
                t_out_pos = atomicAdd(num_groups, 1);
                res.get_column(0).data<Key>()[t_out_pos] = key_data[idx];
                ref.store(t_out_pos, cuda::memory_order_release);
            } 
            else {
                while((t_out_pos = ref.load(cuda::memory_order_acquire)) == unintialized_flag) {}
            }

            // aggregation
            for(gamax::size_type i = 0; i < num_aggs; i++) {
                aggregate(
                    aggs[i].agg_kind,
                    val.get_column(aggs[i].column_index), idx,
                    res.get_column(i+1), t_out_pos);
            }
            
            idx += loop_stride;
        }
    }

    template<typename Key, typename Map>
    __global__ void compute_groupid_kernel(
        Map map_ref,
        Key* keys,
        const gamax::size_type num_items,
        Key* keys_out,
        gamax::size_type* sparse_group_ids,
        gamax::size_type* group_ids_loc,
        gamax::size_type* num_groups)
    {
        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

        while (idx < num_items) {
            auto [slot, is_new_key] = map_ref.insert_and_find(cuco::pair{keys[idx], idx});
            
            if (is_new_key) {
                sparse_group_ids[idx] = atomicAdd(num_groups, 1);
                keys_out[sparse_group_ids[idx]] = keys[idx];
                // TODO: initialize the aggregate results here
            } 
            
            group_ids_loc[idx] = slot->second;
            
            idx += loop_stride;
        }
    }

    __global__ void compute_aggregation(
        gamax::table_device_view val,
        gamax::table_device_view res,
        gamax::size_type* group_ids,
        gamax::groupby::aggregation_request* aggs,
        const gamax::size_type num_aggs) 
    {
        auto num_items = val.num_rows();

        #pragma unroll
        for(int t = get_cuda_tid(); t < num_items; t += nthreads()) {
            #pragma unroll
            for(gamax::size_type i = 0; i < num_aggs; i++) {
                aggregate(
                    aggs[i].agg_kind,
                    val.get_column(aggs[i].column_index), t,
                    res.get_column(i+1), group_ids[t]);
            }
        }
    }

/*
    struct hash_groupby_impl {
        template<typename Key>
        std::unique_ptr<gamax::table> operator()(
            gamax::table& key,
            gamax::table& val,
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
                        estimated_num_groups, 
                        true)); // TODO: fix this
            }
            auto res = std::make_unique<gamax::table>(std::move(final_columns));
            
            constexpr double load_factor = 0.75;
            std::size_t map_capacity = std::ceil(estimated_num_groups / load_factor);
            Key constexpr empty_key_sentinel     = -1; // FIXME: handle negative values and unsigned values correctly 
            gamax::size_type constexpr empty_value_sentinel = -1;            
            auto map = cuco::static_map{map_capacity,
                                    cuco::empty_key{empty_key_sentinel},
                                    cuco::empty_value{empty_value_sentinel},
                                    thrust::equal_to<Key>{},
                                    cuco::linear_probing<1, cuco::default_hash_function<Key>>{}};
            auto map_ref = map.ref(cuco::insert_and_find);

            auto d_num_groups = gamax::make_column(gamax::data_type{size_type_id}, 1, true);

            auto key_view = table_device_view::create(key);
            auto val_view = table_device_view::create(val);
            auto res_view = table_device_view::create(*res);

            hash_groupby_kernel<Key><<<num_tb(input_num_rows, 256), 256, 0, stream>>>(
                map_ref, 
                *key_view, 
                *val_view, 
                *res_view, 
                static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()), 
                agg_requests.size(), 
                d_num_groups->data<gamax::size_type>());
            
            cudaStreamSynchronize(stream);

            gamax::size_type num_groups;
            cudaMemcpy(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost);
            res->set_rows(num_groups);

            return res;
        }
    };
*/
    struct hash_groupby_fast_impl {
        template<typename Key>
        std::unique_ptr<gamax::table> operator()(
            gamax::table& key,
            gamax::table& val,
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
                        estimated_num_groups, 
                        true)); // TODO: fix this
            }
            auto res = std::make_unique<gamax::table>(std::move(final_columns));
            
            constexpr double load_factor = 0.75;
            std::size_t map_capacity = std::ceil(estimated_num_groups / load_factor);
            Key constexpr empty_key_sentinel     = -1; // FIXME: handle negative values and unsigned values correctly 
            gamax::size_type constexpr empty_value_sentinel = -1;            
            auto map = cuco::static_map{map_capacity,
                                    cuco::empty_key{empty_key_sentinel},
                                    cuco::empty_value{empty_value_sentinel},
                                    thrust::equal_to<Key>{},
                                    cuco::linear_probing<1, cuco::default_hash_function<Key>>{}};
            auto map_ref = map.ref(cuco::insert_and_find);

            auto sparse_group_ids = gamax::make_idx_column(input_num_rows, false);
            auto groupid_loc = gamax::make_idx_column(input_num_rows, false);
            auto d_num_groups = gamax::make_column(gamax::data_type{size_type_id}, 1, true);

            compute_groupid_kernel<<<num_tb(input_num_rows, 256), 256, 0, stream>>>(
                map_ref, 
                key.get_column(0).data<Key>(), 
                input_num_rows, 
                res->get_column(0).data<Key>(),
                sparse_group_ids->data<gamax::size_type>(), 
                groupid_loc->data<gamax::size_type>(),
                d_num_groups->data<gamax::size_type>());

            auto group_ids = gamax::make_idx_column(input_num_rows, false);
            gamax::util::device::gather(
                groupid_loc->data<gamax::size_type>(), 
                sparse_group_ids->data<gamax::size_type>(), 
                group_ids->data<gamax::size_type>(), 
                input_num_rows, 
                stream);

            auto val_view = table_device_view::create(val);
            auto res_view = table_device_view::create(*res);

            compute_aggregation<<<num_tb(input_num_rows, 256), 256, 0, stream>>>(
                *val_view, 
                *res_view, 
                group_ids->data<gamax::size_type>(), 
                static_cast<gamax::groupby::aggregation_request*>(d_agg_req_buf->data()), 
                agg_requests.size());
            
            cudaStreamSynchronize(stream);

            gamax::size_type num_groups;
            cudaMemcpy(&num_groups, d_num_groups->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost);
            res->set_rows(num_groups);

            return res;
        }
    };
} // unnamed namespace

// currently only sum and max are supported
std::unique_ptr<gamax::table> groupby(
    gamax::table& key, 
    gamax::table& val, 
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