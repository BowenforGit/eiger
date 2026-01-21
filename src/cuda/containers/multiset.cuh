#pragma once

#include <cuda.h>
#include <memory>
#include <cmath>
#include <cuda/std/tuple>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"
#include "cuco/static_multiset.cuh"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/distance.h"

namespace gamax::containers {

namespace detail {
    /**
     * @brief Hash functor that hashes based on key values at given TID positions.
     * 
     * Used during build phase to hash TIDs based on their corresponding key values.
     * Supports construction from integer (required by double_hashing's Hash2).
     */
    template <typename Key, typename TID>
    struct tid_hasher {
        Key const* keys;
        cuco::default_hash_function<Key> hash_fn{};

        __host__ __device__ tid_hasher(Key const* keys_ptr = nullptr) : keys(keys_ptr) {}
        
        // Required by double_hashing for Hash2 - constructs with a seed
        __host__ __device__ tid_hasher(int seed) : keys(nullptr), hash_fn{} {}

        __device__ auto operator()(TID tid) const {
            return hash_fn(keys[tid]);
        }
    };

    /**
     * @brief Equality functor that compares key values at given TID positions.
     * 
     * Used during build phase to compare two TIDs based on their corresponding key values.
     */
    template <typename Key, typename TID>
    struct tid_equal {
        Key const* keys;

        __host__ __device__ tid_equal(Key const* keys_ptr = nullptr) : keys(keys_ptr) {}

        __device__ bool operator()(TID lhs, TID rhs) const {
            return keys[lhs] == keys[rhs];
        }
    };

    /**
     * @brief Hash functor for probing that hashes probe key at given probe TID.
     * 
     * Takes a probe_tid and returns hash(probe_keys[probe_tid]).
     */
    template <typename Key, typename TID>
    struct probe_tid_hasher {
        Key const* probe_keys;
        cuco::default_hash_function<Key> hash_fn{};

        __host__ __device__ probe_tid_hasher(Key const* probe_keys_ptr = nullptr) : probe_keys(probe_keys_ptr) {}

        __device__ auto operator()(TID probe_tid) const {
            return hash_fn(probe_keys[probe_tid]);
        }
    };

    /**
     * @brief Equality functor for probing that compares probe key against build key.
     * 
     * Takes (probe_tid, build_tid) and returns probe_keys[probe_tid] == build_keys[build_tid].
     */
    template <typename Key, typename TID>
    struct probe_tid_equal {
        Key const* probe_keys;
        Key const* build_keys;

        __host__ __device__ probe_tid_equal(Key const* probe_keys_ptr = nullptr, Key const* build_keys_ptr = nullptr) 
            : probe_keys(probe_keys_ptr), build_keys(build_keys_ptr) {}

        __device__ bool operator()(TID probe_tid, TID build_tid) const {
            return probe_keys[probe_tid] == build_keys[build_tid];
        }
    };
} // namespace detail

    /**
     * @brief GPU hash multiset wrapper for cuco::static_multiset.
     * 
     * This multiset stores only TIDs (tuple IDs), using custom hash and equality functors 
     * that reference the actual key columns. This approach is memory-efficient compared to
     * storing (key, tid) pairs like multimap does.
     * 
     * Build phase: Inserts build TIDs (0, 1, 2, ..., n-1) with hash/equality based on build_keys.
     * Probe phase: Probes with probe TIDs (0, 1, ..., m-1), hash/equality based on probe_keys.
     *              Returns matching (build_tid, probe_tid) pairs.
     * 
     * @tparam Key The actual key type used for comparison and hashing
     * @tparam TID The TID type stored in the multiset (default: gamax::size_type)
     * @tparam Allocator Memory allocator type
     * @tparam CG_SIZE Cooperative group size for cuco probing
     */
    template <typename Key,
              typename TID = gamax::size_type,
              typename Allocator = ThrustAllocator<char>,
              int CG_SIZE = 4>
    class multiset {
    public:
        using tid_hasher_type = detail::tid_hasher<Key, TID>;
        using tid_equal_type = detail::tid_equal<Key, TID>;
        using probing_scheme_type = cuco::double_hashing<CG_SIZE, tid_hasher_type>;

        multiset(
            gamax::column_view const& build_keys,
            double load_factor, // (0.0, 1.0]
            TID empty_tid_sentinel,
            Allocator allocator,
            cudaStream_t stream = cudaStreamDefault)
        : _cuco_allocator(allocator)
        , _load_factor(load_factor)
        , _build_keys_ptr(build_keys.data<Key>())
        , _num_build_keys(build_keys.size())
        , _multiset(
            static_cast<std::size_t>(_num_build_keys * std::ceil(1.0 / load_factor)),
            cuco::empty_key{empty_tid_sentinel},
            tid_equal_type{_build_keys_ptr},
            probing_scheme_type{tid_hasher_type{_build_keys_ptr}, tid_hasher_type{_build_keys_ptr}},
            cuco::cuda_thread_scope<cuda::thread_scope_device>{},
            cuco::storage<2>{},
            _cuco_allocator,
            stream) 
        {
            if(load_factor <= 0.0 || load_factor > 1.0) {
                throw std::invalid_argument("Load factor must be in the range (0.0, 1.0].");
            }
        }
    
        /**
         * @brief Insert TIDs (0, 1, ..., num_build_keys-1) into the multiset.
         * 
         * Only TIDs are stored. The TIDs are hashed and compared based on their 
         * corresponding key values using the build_keys column provided at construction.
         */
        void insert_with_tids(cudaStream_t stream = cudaStreamDefault) 
        {
            auto tids = thrust::make_counting_iterator(static_cast<TID>(0));
            _multiset.insert(tids, tids + _num_build_keys, stream);
            cudaStreamSynchronize(stream);
        }

        /**
         * @brief Insert TIDs for the given keys column into the multiset.
         * 
         * @note The keys parameter is used only for size; insertion uses the 
         *       build_keys from construction. This signature is provided for 
         *       API compatibility with multimap.
         */
        void insert_with_tids(
            gamax::column_view const& keys, 
            cudaStream_t stream = cudaStreamDefault) 
        {
            insert_with_tids(stream);
        }
    
        /**
         * @brief Probe the multiset with probe keys and return matching TID pairs.
         * 
         * @param probe_keys Column of probe keys
         * @param output_size Pre-allocated output size (use count() to determine)
         * @param stream CUDA stream
         * 
         * @return Tuple of (build_tids_out, probe_tids_out) columns
         */
        std::tuple<
            std::unique_ptr<gamax::column>, 
            std::unique_ptr<gamax::column>>
        probe(
            gamax::column_view const& probe_keys,
            gamax::size_type output_size,
            cudaStream_t stream = cudaStreamDefault) 
        {
            auto probe_keys_ptr = probe_keys.data<Key>();
            auto num_probe_keys = probe_keys.size();

            // Probe with probe TIDs (0, 1, ..., num_probe_keys-1)
            auto probe_tids_in = thrust::make_counting_iterator(static_cast<TID>(0));

            // Create output columns
            auto probe_tids_out = gamax::make_column<TID>(output_size);
            auto build_tids_out = gamax::make_column<TID>(output_size);

            // Create probe functors that access probe_keys via pointer
            // Note: double_hashing requires a tuple of two hash functions
            auto probe_hash = cuda::std::make_tuple(
                detail::probe_tid_hasher<Key, TID>{probe_keys_ptr},
                detail::probe_tid_hasher<Key, TID>{probe_keys_ptr});
            auto probe_eq = detail::probe_tid_equal<Key, TID>{probe_keys_ptr, _build_keys_ptr};

            // Retrieve matches: input is probe_tid, outputs are (probe_tid, build_tid)
            auto [probe_tids_out_end, build_tids_out_end] 
            = _multiset.retrieve(
                probe_tids_in,
                probe_tids_in + num_probe_keys,
                probe_eq,
                probe_hash,
                probe_tids_out->template data<TID>(),
                build_tids_out->template data<TID>(),
                stream);
            
            cudaStreamSynchronize(stream);

            auto n_matches = thrust::distance(probe_tids_out->template data<TID>(), probe_tids_out_end);
            probe_tids_out->set_size(n_matches);
            build_tids_out->set_size(n_matches);

            return std::make_tuple(std::move(build_tids_out), std::move(probe_tids_out));
        }

        /**
         * @brief Count the number of matches for probe keys.
         * 
         * @param probe_keys Column of probe keys
         * @param stream CUDA stream
         * 
         * @return Total number of matches
         */
        std::size_t count(
            gamax::column_view const& probe_keys,
            cudaStream_t stream = cudaStreamDefault) 
        {
            auto probe_keys_ptr = probe_keys.data<Key>();
            auto num_probe_keys = probe_keys.size();

            // Probe with probe TIDs
            auto probe_tids_in = thrust::make_counting_iterator(static_cast<TID>(0));

            // Note: double_hashing requires a tuple of two hash functions
            auto probe_hash = cuda::std::make_tuple(
                detail::probe_tid_hasher<Key, TID>{probe_keys_ptr},
                detail::probe_tid_hasher<Key, TID>{probe_keys_ptr});
            auto probe_eq = detail::probe_tid_equal<Key, TID>{probe_keys_ptr, _build_keys_ptr};

            auto n_matches = _multiset.count(
                probe_tids_in,
                probe_tids_in + num_probe_keys,
                probe_eq,
                probe_hash,
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
        double _load_factor;
        Key const* _build_keys_ptr;
        gamax::size_type _num_build_keys;
        cuco::static_multiset<
            TID, 
            cuco::extent<std::size_t>, 
            cuda::thread_scope_device, 
            tid_equal_type, 
            probing_scheme_type, 
            Allocator> _multiset;
    };
} // namespace gamax::containers