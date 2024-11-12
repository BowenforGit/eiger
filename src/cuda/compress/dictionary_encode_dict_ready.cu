#include "dictionary_encode.hpp"

#include <memory>
#include <cuda.h>
#include <cuco/static_map.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <cuda/stream_ref>

namespace gamax::compress {
namespace {
    template<typename Encode>
    struct dict_ready_dict_enc_helper {
        template<typename Key>
        void operator()(
            const gamax::column_view keys, 
            const gamax::column_view dict, 
            gamax::mutable_column_view encoded_keys, 
            cudaStream_t stream) 
        {
            constexpr double load_factor = 0.3;
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
            map.insert(pairs_begin, pairs_begin + dict.size(), cuda::stream_ref{stream});
            map.find(keys.data<Key>(), keys.data<Key>() + keys.size(), encoded_keys.data<Encode>(), cuda::stream_ref{stream});
        }
    };
} // unnamed namespace

std::unique_ptr<gamax::column> dictionary_encode(
    const gamax::column_view column, 
    const gamax::column_view dict,
    const gamax::data_type encoded_type,
    cudaStream_t stream)
{
    auto encoded_col = gamax::make_column(encoded_type, column.size());

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
    const gamax::data_type encoded_type,
    cudaStream_t stream)
{
    return dictionary_encode(column.view(), dict.view(), encoded_type, stream);
}
}
