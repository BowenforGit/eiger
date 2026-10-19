// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "dictionary_encode.hpp"

#include <memory>
#include <cuda.h>
#include <cuco/static_map.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <cuda/stream_ref>

#include "types/sentinel.hpp"

namespace gamax::compress {
namespace {
    template<typename Encode>
    struct dict_ready_dict_enc_helper {
        template<typename Key, std::enable_if_t<!cuco::is_bitwise_comparable_v<Key>, int> = 0>
        void operator()(
            const gamax::column_view keys, 
            const gamax::column_view dict, 
            gamax::mutable_column_view encoded_keys, 
            cudaStream_t stream)
        {
            throw std::runtime_error("Dictionary encoding is not supported for non-integral type.");
        }

        template<typename Key, std::enable_if_t<cuco::is_bitwise_comparable_v<Key>, int> = 0>
        void operator()(
            const gamax::column_view keys, 
            const gamax::column_view dict, 
            gamax::mutable_column_view encoded_keys, 
            cudaStream_t stream) 
        {
            constexpr double load_factor = 0.3;
            gamax::size_type map_capacity = std::ceil(dict.size() / load_factor);
            Key constexpr empty_key_sentinel     = gamax::sentinel<Key>(); // FIXME: handle negative values and unsigned values correctly 
            Encode constexpr empty_value_sentinel = gamax::sentinel<Encode>();    
            auto cuco_allocator = gamax::rmm_allocator_handle<cuco::pair<Key, Encode>>{stream};
            auto map = cuco::static_map{map_capacity,
                                        cuco::empty_key{empty_key_sentinel},
                                        cuco::empty_value{empty_value_sentinel},
                                        thrust::equal_to<Key>{},
                                        cuco::linear_probing<1, cuco::default_hash_function<Key>>{},
                                        {}, {}, 
                                        cuco_allocator.get(),
                                        stream};
            
            auto pairs_begin = thrust::make_zip_iterator(thrust::make_tuple(dict.data<Key>(), thrust::make_counting_iterator<Encode>(0)));
#ifdef __NVCC__
            map.insert(pairs_begin, pairs_begin + dict.size(), cuda::stream_ref{stream});
#elif defined(__HIP_PLATFORM_AMD__)
            auto pairs_it = thrust::make_transform_iterator(pairs_begin, [=] __device__ (auto i) {
                return thrust::make_pair(thrust::get<0>(i), thrust::get<1>(i));
            });
            map.insert(pairs_it, pairs_it + dict.size(), cuda::stream_ref{stream});     
#endif
            map.find(keys.data<Key>(), keys.data<Key>() + keys.size(), encoded_keys.data<Encode>(), cuda::stream_ref{stream});
        }
    };
} // unnamed namespace

std::unique_ptr<gamax::column> dictionary_encode(
    const gamax::column_view column, 
    const gamax::column_view dict,
    cudaStream_t stream,
    const gamax::data_type encoded_type)
{
    auto encoded_col = gamax::make_column(encoded_type, column.size(), stream);

    /* somehow using double_type_dispatcher will not compile, so I have to write it explicitly */
    // if(encoded_type.id() == gamax::type_id::INT16) {
     //   type_dispatcher(column.type(), dict_ready_dict_enc_helper<int16_t>{}, column, dict, *encoded_col, stream);
    // } 
    if(encoded_type.id() == gamax::type_id::INT32) {
        type_dispatcher(column.type(), dict_ready_dict_enc_helper<int32_t>{}, column, dict, encoded_col->mutable_view(), stream);
    } else {
        throw std::runtime_error("Unsupported encoded type");
    }
    cudaStreamSynchronize(stream);
    return encoded_col;
}

std::unique_ptr<gamax::column> dictionary_encode(
    const gamax::column& column, 
    const gamax::column& dict,
    cudaStream_t stream,
    const gamax::data_type encoded_type)
{
    return dictionary_encode(column.view(), dict.view(), stream, encoded_type);
}
}
