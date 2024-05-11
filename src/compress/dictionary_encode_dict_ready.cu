#include "dictionary_encode.hpp"

#include <memory>
#include <cuda.h>
#include <cuco/static_map.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <utils/device_info.hpp>

namespace gamax::compress {
namespace {
template<typename Ref>
    __global__ void dict_ready_dict_enc_hash_kernel(
        Ref map,
        typename Ref::key_type const* const keys,
        const gamax::size_type num_items,
        typename Ref::mapped_type* encoded_keys)
    {
        extern __shared__ int16_t temp[];
        
        auto sm_buffer = reinterpret_cast<typename Ref::window_type *>(temp);
        
        auto g          = cooperative_groups::this_thread_block();
        auto insert_ref = map.make_copy(g, sm_buffer, cuco::thread_scope_block);
        auto find_ref   = std::move(insert_ref).with(cuco::op::find);

        #pragma unroll
        for (size_t i = get_cuda_tid(); i < num_items; i += nthreads())
        {
            auto found_pair_it = find_ref.find(keys[i]);
            encoded_keys[i] = found_pair_it->second;
        }
    }

    template<typename Encode>
    struct dict_ready_dict_enc_helper {
        template<typename Key>
        void operator()(
            const gamax::column& keys, 
            const gamax::column& dict, 
            gamax::column& encoded_keys, 
            cudaStream_t stream) 
        {
            constexpr double load_factor = 0.75;
            gamax::size_type map_capacity = std::ceil(dict.size() / load_factor);
            Key constexpr empty_key_sentinel     = -1; // FIXME: handle negative values and unsigned values correctly 
            Encode constexpr empty_value_sentinel = -1;    
            auto cuco_allocator = *get_thrust_allocator<cuco::pair<Key, Encode>>(stream);   
            auto map = cuco::static_map{map_capacity,
                                        cuco::empty_key{empty_key_sentinel},
                                        cuco::empty_value{empty_value_sentinel},
                                        thrust::equal_to<Key>{},
                                        cuco::linear_probing<1, cuco::default_hash_function<Key>>{},
                                        {}, {}, cuco_allocator};
            
            auto pairs_begin = thrust::make_zip_iterator(thrust::make_tuple(dict.data<Key>(), thrust::make_counting_iterator<Encode>(0)));
            map.insert(pairs_begin, pairs_begin + dict.size(), cuco::cuda_stream_ref{stream});
            auto map_ref = map.ref(cuco::insert);

            auto fn = dict_ready_dict_enc_hash_kernel<decltype(map_ref)>;

            size_t shared_mem_size = sizeof(typename decltype(map_ref)::window_type) * map_ref.capacity();
            if(shared_mem_size > 48 * 1024 && shared_mem_size <= 96 * 1024) {
                CHECK_CUDA_ERROR(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
            } else if(shared_mem_size > 96 * 1024) {
                throw std::runtime_error("Too much shared memory required for the kernel");
            }

            int block_size = 256;
            int maxActiveBlocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxActiveBlocks, fn, block_size, shared_mem_size);
            int grid_size = SM_COUNT * maxActiveBlocks;
            fn<<<grid_size, block_size, shared_mem_size, stream>>>(map_ref, keys.data<Key>(), keys.size(), encoded_keys.data<Encode>());
        }
    };
} // unnamed namespace

std::unique_ptr<gamax::column> dictionary_encode(
    const gamax::column& column, 
    const gamax::column& dict,
    const gamax::data_type encoded_type,
    cudaStream_t stream)
{
    auto encoded_col = gamax::make_column(encoded_type, column.size());
    // double_type_dispatcher(column.type(), encoded_type, dict_ready_dict_enc_helper{}, column, dict, *encoded_col, stream);
    if(encoded_type.id() == gamax::type_id::INT16) {
        type_dispatcher(column.type(), dict_ready_dict_enc_helper<int16_t>{}, column, dict, *encoded_col, stream);
    } else if(encoded_type.id() == gamax::type_id::INT32) {
        type_dispatcher(column.type(), dict_ready_dict_enc_helper<int32_t>{}, column, dict, *encoded_col, stream);
    } else {
        throw std::runtime_error("Unsupported encoded type");
    }
    return encoded_col;
}
}
