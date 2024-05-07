#pragma once

#include <unordered_map>
#include <list>
#include <memory>
#include <cassert>

#include "utils/check_error.hpp"

// template<size_t kInitialPoolSize, size_t kAlignBits = 256UL>
class UserSpaceMM {
private:
    using block_type = struct { char* data {nullptr}; size_t sz {0}; };

public:
    UserSpaceMM(const size_t kInitialPoolSize, const size_t kAlignBits = 256UL);

    ~UserSpaceMM();

    void allocate(void** ptr, bool clear, size_t sz, cudaStream_t stream);

    void release(void* ptr, cudaStream_t stream);

    size_t get_peak_mem_used() const;

    size_t get_mem_used() const;

private:
    // find a block in the free list that is at least as large as sz
    block_type find_block(const size_t sz);

    // Insert the block back to the free list, merge if necessary
    void insert_block(const block_type& free);

    size_t roundup(const size_t value);

    char* roundup(const char* ptr);

    bool can_alloc(const block_type& block, const size_t sz);

private:
    std::unordered_map<void*, block_type> alloc_map;
    std::list<block_type> free_list;
    void* root{nullptr};
    size_t mem_used {0};
    size_t peak_mem_used {0};
    size_t align_bits;
};

template<typename T>
class ThrustAllocator {
public:
    template<typename U>
    friend class ThrustAllocator;

    using value_type = T;

    ThrustAllocator() = default;

    ThrustAllocator(std::shared_ptr<UserSpaceMM> mm, cudaStream_t stream) : mm_(mm), stream_(stream) {}

    T* allocate(std::size_t num_items) {
        T* ptr;
        mm_->allocate((void**)&ptr, false, num_items*sizeof(T), stream_);
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) {
        mm_->release(ptr, stream_);
    }

    template <class U>
    ThrustAllocator(ThrustAllocator<U> const& other) noexcept
    {
        mm_ = other.mm_;
        stream_ = other.stream_;
    }

private:
    std::shared_ptr<UserSpaceMM> mm_;
    cudaStream_t stream_;
};

constexpr size_t mem_pool_size = 25769803776*0.9;
// extern UserSpaceMM* mm;
// extern ThrustAllocator<char>* thrust_allocator;

std::shared_ptr<UserSpaceMM> get_mm();
void delete_mm();

template<typename T = char>
std::unique_ptr<ThrustAllocator<T>> get_thrust_allocator(cudaStream_t stream = cudaStreamDefault) {
    return std::make_unique<ThrustAllocator<T>>(get_mm(), stream);
}

namespace gamax::details {
    void alloc_by_cuda(void** ptr, bool clear, size_t sz, cudaStream_t stream);
    size_t get_cuda_free_mem();
    void alloc_by_rmm_mempool(void** ptr, bool clear, size_t sz, cudaStream_t stream);
    void free_rmm_mempool(void* ptr, cudaStream_t stream);
} // gamax::details

// #define USE_CUDA_MEMALLOC
template<typename T>
void allocate_mem(T** ptr, bool clear = true, size_t sz = sizeof(T), cudaStream_t stream = 0) {
    assert(ptr != nullptr);

#ifdef USE_CUDA_MEMALLOC
    gamax::details::alloc_by_cuda((void**)ptr, clear, sz, stream);
#else
    gamax::details::alloc_by_rmm_mempool((void**)ptr, clear, sz, stream);
#endif
}

template<typename T>
void release_mem(T* ptr, cudaStream_t stream = 0) {
#ifdef USE_CUDA_MEMALLOC
    CHECK_CUDA_ERROR(cudaFreeAsync(ptr, stream));
#else
    gamax::details::free_rmm_mempool((void*)ptr, stream);
#endif
}

void release_all_mem();
void debug_cuda_mem_usage();