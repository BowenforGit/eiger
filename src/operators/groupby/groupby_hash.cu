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
#include "agg_helper.cuh"

namespace gamax::groupby::hash {
namespace {
    using namespace gamax::groupby::details;

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
            auto cuco_allocator = *get_thrust_allocator<cuco::pair<Key, gamax::size_type>>(stream);   
            auto map = cuco::static_map{map_capacity,
                                    cuco::empty_key{empty_key_sentinel},
                                    cuco::empty_value{empty_value_sentinel},
                                    thrust::equal_to<Key>{},
                                    cuco::linear_probing<1, cuco::default_hash_function<Key>>{}
                                    , {}, {},
                                    cuco_allocator
                                    };
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