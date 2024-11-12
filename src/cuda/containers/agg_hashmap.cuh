#pragma once

#include <cuda.h>
#include <limits>

#include "atomics/atomics.cuh"

namespace gamax {
template<typename KeyT, typename ValueT>
class AggHashMapEntries {
public:
    KeyT* keys {nullptr};
    ValueT* values {nullptr};
    int* size {nullptr};

    __device__ __forceinline__ AggHashMapEntries(void* data, const int capacity) {
        keys = reinterpret_cast<KeyT*>(data);
        values = reinterpret_cast<ValueT*>(&keys[capacity]);
        size = reinterpret_cast<int*>(&values[capacity]);
    }

    __device__ __forceinline__ void inc(int n) {
        atomicAdd(size, n);
    }

    __device__ __forceinline__ void dec(int n) {
        atomicSub(size, n);
    }

    __device__ __forceinline__ void set_size(int sz) const {
        *size = sz;
    }

    __device__ __forceinline__ int get_size() const {
        return *size;
    }
};

/**
 * A hash map for the shared memory
*/
template<typename KeyT, 
         typename ValueT, 
         typename HashFn, 
         typename UponInsertFn,
         KeyT kSentinel = std::numeric_limits<KeyT>::min()>
class AggHashMap {
    static_assert(UponInsertFn::is_monoid && UponInsertFn::is_atomic_updatable);

public:
    enum ActionStatus {
        SUCCESS=0,
        FAIL
    };

    __device__ AggHashMap(void* data, const int capacity, HashFn hash_fn)
    : capacity_(capacity)
    , entries_(data, capacity)
    , hash_fn_(hash_fn) {}

    __device__ __forceinline__ void Init() {
        if(threadIdx.x == 0) {
            entries_.set_size(0);
        }
        for (int i = threadIdx.x; i < capacity_; i+=blockDim.x) {
            entries_.keys[i] = kSentinel;
            entries_.values[i] = UponInsertFn::identity;
        }
        __syncthreads();
    }

    /**
     * Insert a key-value pair into the hash map
     * if the key already exists, the value will be updated by applying the upon_insert_fn_;
     * otherwise, the key-value pair will be inserted into the hash map
    */
    __device__ __forceinline__ ActionStatus Insert(const KeyT& key, const ValueT& value) {
        if (entries_.get_size() == capacity_) {
            return ActionStatus::FAIL;
        }
        
        auto index = hash_fn_(key) % capacity_;
        KeyT old;
        while((old = gamax::atomic::atomic_cas(&entries_.keys[index], kSentinel, key)) != kSentinel) {
            if (entries_.keys[index] == key) {
                break;
            }
            index = (index + 1) % capacity_;
        }
        // post-loop condition: entries_.keys[index] == key
        
        if(old == kSentinel) entries_.inc(1);
        upon_insert_fn_.atomic_update(&entries_.values[index], value);
        return ActionStatus::SUCCESS;
    }
    
    __device__ __forceinline__ int size() const {
        return entries_.get_size();
    }

    __device__ __forceinline__ int capacity() const {
        return capacity_;
    }

    __device__ __forceinline__ double load_factor() const {
        return static_cast<double>(entries_.get_size()) / capacity_;
    }

    __device__ __forceinline__ std::pair<KeyT, ValueT> operator[](int index) const {
        return std::make_pair(entries_.keys[index], entries_.values[index]);
    }

    __device__ __forceinline__ bool is_empty(int index) const {
        return entries_.keys[index] == kSentinel;
    }

private:
    using HashMapEntriesT = AggHashMapEntries<KeyT, ValueT>;
    HashMapEntriesT entries_;
    int capacity_;
    HashFn hash_fn_;
    UponInsertFn upon_insert_fn_;
};
} // namespace gamax