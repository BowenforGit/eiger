// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cuda.h>
#include <memory>
#include <cmath>
#include <cuda/std/tuple>

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
              typename Allocator = rmm_cuco_allocator<cuco::pair<Key, Value>>,
              int CG_SIZE = 2,
              typename KeyEqual = cuda::std::equal_to<Key>,
              typename ProbingScheme = cuco::double_hashing<CG_SIZE, cuco::default_hash_function<Key>>>
    class multimap {
    public:
        multimap(
            gamax::size_type capacity,
            double load_factor, // (0.0, 1.0]
            Key empty_key_sentinel,
            Value empty_value_sentinel,
            Allocator allocator,
            cudaStream_t stream)
        : _cuco_allocator(allocator)
        , _load_factor(load_factor)
        , _multimap(
            capacity,
            load_factor,
            cuco::empty_key{empty_key_sentinel}, 
            cuco::empty_value{empty_value_sentinel},
            KeyEqual{},
            ProbingScheme{},
            cuco::cuda_thread_scope<cuda::thread_scope_device>{},
            cuco::storage<2>{},
            _cuco_allocator,
            stream) {
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
            cudaStream_t stream)
        {
            auto probe_tids_in = thrust::make_counting_iterator(static_cast<Value>(0));
            auto probe_kv_pairs_in = thrust::make_zip_iterator(cuda::std::make_tuple(probe_keys.data<Key>(), probe_tids_in));
            
            auto probe_keys_out = gamax::make_column<Key>(output_size, stream);
            auto probe_tids_out = gamax::make_column<Value>(output_size, stream);
            auto probe_kv_pairs_out = thrust::make_zip_iterator(cuda::std::make_tuple(probe_keys_out->template data<Key>(), probe_tids_out->template data<Value>()));

            auto build_keys_out = gamax::make_column<Key>(output_size, stream);
            auto build_tids_out = gamax::make_column<Value>(output_size, stream);
            auto build_kv_pairs_out = thrust::make_zip_iterator(cuda::std::make_tuple(build_keys_out->template data<Key>(), build_tids_out->template data<Value>()));

            auto pair_equal = [] __device__ (auto const& a, auto const& b) {
                return a.first == b.first;
            };

            auto [probe_kv_pairs_out_end, build_kv_pairs_out_end] 
            =  _multimap.pair_retrieve(
                probe_kv_pairs_in, 
                probe_kv_pairs_in + probe_keys.size(), 
                probe_kv_pairs_out,
                build_kv_pairs_out,
                pair_equal,
                stream);
            
            cudaStreamSynchronize(stream);

            auto n_matches = cuda::std::distance(probe_kv_pairs_out, probe_kv_pairs_out_end);
            probe_keys_out->set_size(n_matches);
            probe_tids_out->set_size(n_matches);
            build_keys_out->set_size(n_matches);
            build_tids_out->set_size(n_matches);

            return std::make_tuple(std::move(probe_keys_out), std::move(build_tids_out), std::move(probe_tids_out));
        }

        std::size_t count(
            gamax::column_view const& probe_keys,
            cudaStream_t stream)
        {
            auto probe_tids_in = thrust::make_counting_iterator(static_cast<Value>(0));
            auto probe_kv_pairs_in = thrust::make_zip_iterator(cuda::std::make_tuple(probe_keys.data<Key>(), probe_tids_in));

            auto pair_equal = [] __device__ (auto const& a, auto const& b) {
                return a.first == b.first;
            };

            auto n_matches = _multimap.pair_count(
                probe_kv_pairs_in,
                probe_kv_pairs_in + probe_keys.size(),
                pair_equal,
                stream);
            
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
        cuco::static_multimap<
            Key,
            Value,
            cuco::extent<std::size_t>,
            cuda::thread_scope_device,
            KeyEqual,
            ProbingScheme,
            Allocator> _multimap;
        double _load_factor;
    };
    } // namespace gamax::containers
