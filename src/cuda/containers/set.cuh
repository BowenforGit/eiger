#pragma once

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"
#include "cuco/static_set.cuh"

namespace gamax::containers {
template <typename Key,
          typename Value = Key, // used for the mapping table case 
          typename KeyEqual = thrust::equal_to<Value>,
          typename HashFn = cuco::default_hash_function<Value>,
          typename ProbingScheme = cuco::double_hashing<4, HashFn>>
class set {
public:
    set(
        gamax::size_type capacity,
        double load_factor,
        Key empty_key_sentinel,
        cudaStream_t stream = cudaStreamDefault,
        KeyEqual key_equal = KeyEqual{},
        HashFn hash_fn = HashFn{})
    : _cuco_allocator(get_thrust_allocator<Key>(stream))
    , _set(
        capacity, 
        load_factor, 
        cuco::empty_key{empty_key_sentinel}, 
        key_equal, 
        ProbingScheme{hash_fn}, 
        cuco::thread_scope_device, 
        cuco::storage<1>{}, 
        *_cuco_allocator, 
        stream) {}

    void insert(gamax::column_view const& keys, cudaStream_t stream = cudaStreamDefault) {
        auto keys_ptr = keys.data<Key>();
        auto num_keys = keys.size();
        _set.insert(keys_ptr, keys_ptr+num_keys, stream); // this synchronizes the stream
    }

    std::unique_ptr<gamax::column> contains(gamax::column_view const& keys, cudaStream_t stream = cudaStreamDefault) {
        auto keys_ptr = keys.data<Key>();
        auto num_keys = keys.size();
        auto result = gamax::make_column<bool>(num_keys);
        _set.contains(keys_ptr, keys_ptr+num_keys, result->data<bool>(), stream); // this synchronizes the stream
        return result;
    }

private:
    std::unique_ptr<ThrustAllocator<Key>> _cuco_allocator;
    cuco::static_set<
        Key, 
        cuco::extent<std::size_t>, 
        cuda::thread_scope_device, 
        KeyEqual, 
        ProbingScheme, 
        ThrustAllocator<Key>, 
        cuco::storage<1>> _set;
};
} // namespace gamax::containers