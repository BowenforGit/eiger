#include "compress/dictionary_encode.hpp"

#include <memory>
#include <stdexcept>
#include <cmath>
#include <cuco/static_map.cuh>
#include <thrust/functional.h>
#include <cuda.h>
#include <cuda/std/atomic>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "types/sentinel.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "utils/utils.cuh"
#include "atomics/atomics.cuh"

namespace gamax::compress {
namespace {
    void check_dictionary_encode_solvable(const gamax::column& column) {
        if(gamax::size_of(column.type()) != 4 && gamax::size_of(column.type()) != 8) {
            throw std::runtime_error("Cannot dictionary encode column of type " + gamax::to_string(column.type()));
        }
    }

    void check_dictionary_encode_solvable(const gamax::table& table) {
        throw std::runtime_error("Cannot dictionary encode table");
    }

    template<typename T, typename Map, typename EncodedT = gamax::size_type>
    __global__ void dictionary_encode_kernel(
        Map map_ref, 
        const T* data, 
        const gamax::size_type num_items, 
        EncodedT* encoded_data, 
        gamax::size_type* dict_size) 
    {
        int64_t const loop_stride = gridDim.x * blockDim.x;
        int64_t idx               = blockDim.x * blockIdx.x + threadIdx.x;

        EncodedT unintialized_flag = gamax::sentinel<EncodedT>()-1; // TODO: not the best solution
        while (idx < num_items) {
            auto [slot, is_new_key] = map_ref.insert_and_find(cuco::pair{data[idx], unintialized_flag});
            auto ref = cuda::atomic_ref<EncodedT, cuda::thread_scope_device>{slot->second};
            
            EncodedT encoded_value{0};
            if (is_new_key) {
                encoded_value = gamax::atomic::atomic_add(dict_size, static_cast<gamax::size_type>(1));
                ref.store(encoded_value, cuda::memory_order_release);
            } 
            else {
                while((encoded_value = ref.load(cuda::memory_order_acquire)) == unintialized_flag) {}
            }
            encoded_data[idx] = encoded_value;
            
            idx += loop_stride;
        }
    }

    struct dictionary_encode_impl {
        template<typename T>
        std::pair<std::unique_ptr<gamax::column>, gamax::size_type> operator()(
            const gamax::column& column, 
            const gamax::size_type dictionary_size_hint,
            const double load_factor,
            cudaStream_t stream) 
        {
            using encode_t = gamax::size_type;
            
            auto map_capacity = dictionary_size_hint == 0 ? column.size() : dictionary_size_hint;
            map_capacity = std::ceil(map_capacity / load_factor);
            T constexpr empty_key_sentinel     = gamax::sentinel<T>(); // FIXME: handle negative values and unsigned values correctly 
            encode_t constexpr empty_value_sentinel = gamax::sentinel<encode_t>();            
            auto map = cuco::static_map{map_capacity,
                                    cuco::empty_key{empty_key_sentinel},
                                    cuco::empty_value{empty_value_sentinel},
                                    thrust::equal_to<T>{},
                                    cuco::linear_probing<1, cuco::default_hash_function<T>>{}};
            auto map_ref = map.ref(cuco::insert_and_find);
            
            auto encoded_col = gamax::make_idx_column(column.size());
            auto d_dict_size = gamax::make_idx_column(1, true);

            dictionary_encode_kernel<<<num_tb(column.size(), 256), 256, 0, stream>>>(map_ref, column.data<T>(), column.size(), encoded_col->data<encode_t>(), d_dict_size->data<gamax::size_type>());
            
            gamax::size_type dict_size;
            cudaMemcpy(&dict_size, d_dict_size->data<gamax::size_type>(), sizeof(gamax::size_type), cudaMemcpyDeviceToHost);

            cudaStreamSynchronize(stream);

            return std::make_pair(std::move(encoded_col), dict_size);
        }
    };
}

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::column& column, 
    const gamax::size_type dictionary_size_hint,
    const double load_factor,
    cudaStream_t stream)
{
    check_dictionary_encode_solvable(column);

    return type_dispatcher(
        column.type(), 
        dictionary_encode_impl{}, 
        column, 
        dictionary_size_hint, 
        load_factor, 
        stream);
}

std::pair<std::unique_ptr<gamax::column>, gamax::size_type> dictionary_encode(
    const gamax::table& table, 
    const gamax::size_type dictionary_size_hint,
    const double load_factor,
    cudaStream_t stream) 
{
    check_dictionary_encode_solvable(table);
    return std::make_pair(nullptr, 0);
}
}