#include <cuda.h>
#include <nvbench/nvbench.cuh>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include "memory/column.hpp"
#include "utils/utils.cuh"
#include "compress/dictionary_encode.hpp"

template<typename Key, typename EncodeIt, typename Encode = typename std::iterator_traits<EncodeIt>::value_type>
__global__ void dict_ready_dict_enc_kernel(
    Key* keys,
    Key* dict,
    const gamax::size_type num_items,
    const gamax::size_type dict_size,
    EncodeIt itr,
    Encode* encoded_keys)
{
    extern __shared__ int16_t temp[];
    Key* dict_keys = reinterpret_cast<Key*>(temp);

    // load the dict into the shared memory
    #pragma unroll
    for (size_t i = threadIdx.x; i < dict_size; i += blockDim.x)
    {
        dict_keys[i] = dict[i];
    }
    __syncthreads();

    // use binary search to find the key in the dict
    #pragma unroll
    for (size_t i = get_cuda_tid(); i < num_items; i += nthreads())
    {
        Key key = keys[i];
        
        int low = 0, high = dict_size - 1;
        while(low <= high) {
            int mid = (low + high) / 2;
            if(dict_keys[mid] < key) {
                low = mid + 1;
            } else if(dict_keys[mid] > key) {
                high = mid - 1;
            } else {
                encoded_keys[i] = itr[mid];
                break;
            }
        }
    }
}

template<typename Key, typename Encode>
void bench_dict_ready_dict_enc(nvbench::state& state, nvbench::type_list<Key, Encode>) 
{
    release_all_mem();

    gamax::size_type num_items = state.get_int64("num_items");
    gamax::size_type dict_size = state.get_int64("dict_size");

    auto dict = gamax::make_sequence<Key>(dict_size, 0, 1);
    auto keys = gamax::make_sequence<Key>(num_items, 0, 1);

    auto raw_key_ptr = keys->template data<Key>();
    
    thrust::for_each_n(thrust::device, raw_key_ptr, num_items, [=] __device__ (Key& key) { key = key % dict_size; });
    thrust::default_random_engine g;
    thrust::shuffle(thrust::device, raw_key_ptr, raw_key_ptr + num_items, g);
    cudaDeviceSynchronize();

    state.set_cuda_stream(nvbench::make_cuda_stream_view(cudaStreamDefault));
    state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { 
                auto res = gamax::compress::dictionary_encode(*keys, *dict, gamax::data_type{gamax::type_to_id<Encode>()}, launch.get_stream());
    });
}

NVBENCH_BENCH_TYPES(bench_dict_ready_dict_enc,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>, 
                                      nvbench::type_list<int16_t, int32_t>))
  .set_name("dict_ready_dict_enc")
  .set_type_axes_names({"Key", "Encode"})
  .add_int64_power_of_two_axis("num_items", {27, 28, 29})
  .add_int64_power_of_two_axis("dict_size", nvbench::range(2, 14, 2));