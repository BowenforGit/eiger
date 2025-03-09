#pragma once

#include <cuda.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"
#include "cuco/static_multimap.cuh"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/zip_iterator.h"

namespace gamax::containers {
    template <typename Key,
              typename Value = gamax::size_type,  
              typename Allocator = ThrustAllocator<char>>
    class multimap {
    public:
        multimap(
            gamax::size_type capacity,
            // double load_factor,
            Key empty_key_sentinel,
            Value empty_value_sentinel,
            Allocator allocator,
            cudaStream_t stream = cudaStreamDefault)
        : _cuco_allocator(allocator)
        , _multimap(
            capacity, 
            // load_factor, 
            cuco::empty_key{empty_key_sentinel}, 
            cuco::empty_value{empty_value_sentinel},
            stream,
            _cuco_allocator) {}
    
        void insert_with_tids(
            gamax::column_view const& keys, 
            cudaStream_t stream = cudaStreamDefault) 
        {
            auto keys_ptr = keys.data<Key>();
            auto tids = thrust::make_counting_iterator(0);
            auto kv_pairs = thrust::make_zip_iterator(thrust::make_tuple(keys_ptr, tids));
            auto num_keys = keys.size();
            _multimap.insert(kv_pairs, kv_pairs+num_keys, stream);
            cudaStreamSynchronize(stream);
        }
    
        std::tuple<
            std::unique_ptr<gamax::column>, 
            std::unique_ptr<gamax::column>, 
            std::unique_ptr<gamax::column>>
        probe(
            gamax::column_view const& probe_keys,
            gamax::size_type output_size,
            cudaStream_t stream = cudaStreamDefault) 
        {
            auto probe_tids_in = thrust::make_counting_iterator(0);
            auto probe_kv_pairs_in = thrust::make_zip_iterator(thrust::make_tuple(probe_keys.data<Key>(), probe_tids_in));
            
            auto probe_keys_out = gamax::make_column<Key>(output_size);
            auto probe_tids_out = gamax::make_column<Value>(output_size);
            auto probe_kv_pairs_out = thrust::make_zip_iterator(thrust::make_tuple(probe_keys_out->template data<Key>(), probe_tids_out->template data<Value>()));

            auto build_keys_out = gamax::make_column<Key>(output_size);
            auto build_tids_out = gamax::make_column<Value>(output_size);
            auto build_kv_pairs_out = thrust::make_zip_iterator(thrust::make_tuple(build_keys_out->template data<Key>(), build_tids_out->template data<Value>()));

            auto pair_equal = [] __device__ (auto const& a, auto const& b) {
                return a.first == b.first;
            };

            auto [probe_kv_pairs_out_end, build_kv_pairs_out_end] 
            =  _multimap.pair_retrieve(
                probe_kv_pairs_in, 
                probe_kv_pairs_in+output_size, 
                probe_kv_pairs_out,
                build_kv_pairs_out,
                pair_equal,
                stream);
            
            cudaStreamSynchronize(stream);

            auto n_matches = thrust::distance(probe_kv_pairs_out, probe_kv_pairs_out_end);
            probe_keys_out->set_size(n_matches);
            probe_tids_out->set_size(n_matches);
            build_keys_out->set_size(n_matches);
            build_tids_out->set_size(n_matches);

            return std::make_tuple(std::move(probe_keys_out), std::move(build_tids_out), std::move(probe_tids_out));
        }
    
    private:
        Allocator _cuco_allocator;
        cuco::static_multimap<Key, Value, cuda::thread_scope_device, Allocator> _multimap;
    };
    } // namespace gamax::containers