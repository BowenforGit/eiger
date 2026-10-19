// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cuda.h>
#include <memory>
#include <cmath>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"
#include "cuco/static_map.cuh"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/zip_iterator.h"

namespace gamax::containers {
    template <typename Key,
              typename Value = gamax::size_type,  
              int CG_SIZE = 2,
              typename Allocator = rmm_cuco_allocator<cuco::pair<Key, Value>>,
              typename KeyEqual = cuda::std::equal_to<Key>, 
              typename ProbingScheme = cuco::linear_probing<CG_SIZE, cuco::default_hash_function<Key>>>
    class map {
    public:
        map(
            gamax::size_type capacity,
            double load_factor, // (0.0, 1.0]
            Key empty_key_sentinel,
            Value empty_value_sentinel,
            Allocator allocator,
            cudaStream_t stream)
        : _cuco_allocator(allocator)
        , _load_factor(load_factor)
        , _map(
            capacity,
            load_factor,
            cuco::empty_key{empty_key_sentinel}, 
            cuco::empty_value{empty_value_sentinel},
            {},{},{},{},
            _cuco_allocator,
            stream)
        {
            if(load_factor <= 0.0 || load_factor > 1.0) {
                throw std::invalid_argument("Load factor must be in the range (0.0, 1.0].");
            }
        }
    
        void insert_with_tids(
            gamax::column_view const& keys, 
            cudaStream_t stream)
        {
            auto keys_ptr = keys.data<Key>();
            auto tids = thrust::make_counting_iterator(static_cast<Value>(0));
            auto kv_pairs = thrust::make_zip_iterator(cuda::std::make_tuple(keys_ptr, tids));
            auto num_keys = keys.size();
            _map.insert(kv_pairs, kv_pairs+num_keys, stream);
            cudaStreamSynchronize(stream);
        }

        std::size_t count(
            gamax::column_view const& probe_keys,
            cudaStream_t stream)
        {
            auto keys_ptr = probe_keys.data<Key>();
            auto num_keys = probe_keys.size();
            auto n_matches = _map.count(keys_ptr, keys_ptr + num_keys, stream);
            
            return n_matches;
        }

        constexpr static int cg_size() {
            return CG_SIZE;
        }

        double load_factor() const {
            return _load_factor;
        }
    
    private:
        Allocator _cuco_allocator;
        cuco::static_map<
            Key, 
            Value,
            cuco::extent<std::size_t>, 
            cuda::thread_scope_device, 
            KeyEqual, 
            ProbingScheme, 
            Allocator> _map;
        double _load_factor;
    };
    } // namespace gamax::containers
