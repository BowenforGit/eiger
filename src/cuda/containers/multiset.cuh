#pragma once

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"
#include "cuco/static_multiset.cuh"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/distance.h"

namespace gamax::containers {
    template <typename Key,
              typename Allocator = ThrustAllocator<char>>
    class multiset {
    public:
        multiset(
            gamax::size_type capacity,
            Key empty_key_sentinel,
            Allocator allocator,
            cudaStream_t stream = cudaStreamDefault)
        : _cuco_allocator(allocator)
        , _multiset(
            capacity, 
            cuco::empty_key{empty_key_sentinel}, 
            stream,
            _cuco_allocator) {}
    
        void insert_with_tids(
            gamax::column_view const& keys, 
            cudaStream_t stream = cudaStreamDefault) 
        {
            auto keys_ptr = keys.data<Key>();
            auto num_keys = keys.size();
            _multiset.insert(keys_ptr, keys_ptr + num_keys, stream);
            cudaStreamSynchronize(stream);
        }
    
        std::tuple<
            std::unique_ptr<gamax::column>, 
            std::unique_ptr<gamax::column>>
        probe(
            gamax::column_view const& probe_keys,
            gamax::size_type output_size,
            cudaStream_t stream = cudaStreamDefault) 
        {
            auto probe_keys_out = gamax::make_column<Key>(output_size);
            auto match_keys_out = gamax::make_column<Key>(output_size);

            auto [probe_keys_out_end, match_keys_out_end] 
            = _multiset.retrieve(
                probe_keys.data<Key>(), 
                probe_keys.data<Key>() + probe_keys.size(), 
                probe_keys_out->template data<Key>(),
                match_keys_out->template data<Key>(),
                stream);
            
            cudaStreamSynchronize(stream);

            auto n_matches = thrust::distance(probe_keys_out->template data<Key>(), probe_keys_out_end);
            probe_keys_out->set_size(n_matches);
            match_keys_out->set_size(n_matches);

            return std::make_tuple(std::move(probe_keys_out), std::move(match_keys_out));
        }
    
    private:
        Allocator _cuco_allocator;
        cuco::static_multiset<Key, cuco::extent<std::size_t>, cuda::thread_scope_device, cuda::std::equal_to<Key>, cuco::double_hashing<4, cuco::default_hash_function<Key>>, Allocator> _multiset;
    };
    } // namespace gamax::containers