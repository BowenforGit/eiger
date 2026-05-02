#pragma once

#include <unordered_map>
#include <list>
#include <memory>
#include <cassert>
#include <cuda_runtime_api.h>
#include <cuda/stream_ref>

#include "utils/check_error.hpp"

class MemoryManager {
public:
    virtual void allocate(void** ptr, bool clear, size_t sz, cudaStream_t stream) = 0;
    virtual void release(void* ptr, cudaStream_t stream) = 0;
    size_t get_peak_mem_used() const { return peak_mem_used; }
    size_t get_mem_used() const { return mem_used; }
    // Reset the peak high-water mark to the *current* in-flight bytes.
    // Use this to bracket a region of work — call before, then read
    // `get_peak_mem_used()` after to obtain the high-water mark
    // observed within the bracket. (Lower-bounded at `mem_used` rather
    // than zero so the post-bracket peak is always >= the bracket's
    // starting baseline.)
    void reset_peak_mem_used() { peak_mem_used = mem_used; }

protected:
    size_t mem_used {0};
    size_t peak_mem_used {0};
};

class UserSpaceMM : public MemoryManager {
private:
    using block_type = struct { char* data {nullptr}; size_t sz {0}; };

public:
    UserSpaceMM(const size_t kInitialPoolSize, const size_t kAlignBits = 256UL);

    ~UserSpaceMM();

    void allocate(void** ptr, bool clear, size_t sz, cudaStream_t stream) override;

    void release(void* ptr, cudaStream_t stream) override;

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
    size_t align_bits;
};

class CudaMM : public MemoryManager {
public:
    void allocate(void** ptr, bool clear, size_t sz, cudaStream_t stream) override;
    void release(void* ptr, cudaStream_t stream) override;

private:
    std::unordered_map<void*, std::size_t> alloc_map;
};

template<typename T>
class ThrustAllocator {
public:
    template<typename U>
    friend class ThrustAllocator;

    using value_type = T;

    ThrustAllocator() = default;

    ThrustAllocator(std::shared_ptr<MemoryManager> mm, cudaStream_t stream) : mm_(mm), stream_(stream) {}

    T* allocate(std::size_t num_items) {
        T* ptr;
        mm_->allocate((void**)&ptr, false, num_items*sizeof(T), stream_);
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) {
        mm_->release(ptr, stream_);
    }

    T* allocate(std::size_t num_items, cuda::stream_ref stream) {
        T* ptr;
        mm_->allocate((void**)&ptr, false, num_items*sizeof(T), stream.get());
        return ptr;
    }

    void deallocate(T* ptr, std::size_t, cuda::stream_ref stream) {
        mm_->release(ptr, stream.get());
    }


    template <class U>
    ThrustAllocator(ThrustAllocator<U> const& other) noexcept
    {
        mm_ = other.mm_;
        stream_ = other.stream_;
    }

private:
    std::shared_ptr<MemoryManager> mm_;
    cudaStream_t stream_;
};

// Switch the global memory manager to use cudaMallocAsync /
// cudaFreeAsync directly (CudaMM) instead of the UserSpaceMM slab
// pool. Must be called before any allocation — i.e. before the
// first get_mm() — otherwise throws.
//
// Why: UserSpaceMM cudaMalloc's ~95% of free GPU memory up front and
// hands out aligned slices from the slab. compute-sanitizer / memcheck
// only sees the one big slab as a legal allocation, so an OOB write
// inside the slab boundary looks valid to the tool. Switching to
// CudaMM makes each `cudaMallocAsync` a separately-tracked allocation
// the sanitizer can detect overflows on.
//
// The default (no call to use_cuda_mm) keeps the existing UserSpaceMM
// behaviour. The previous compile-time `USE_DEFAULT_MEMALLOC` macro
// is removed in favour of this runtime switch — debug binaries can
// just call `use_cuda_mm()` at start of main() without rebuilding the
// library.
//
// Idempotent: calling use_cuda_mm() multiple times before first
// allocation is a no-op.
void use_cuda_mm();

std::shared_ptr<MemoryManager> get_mm();

template<typename T = char>
std::unique_ptr<ThrustAllocator<T>> get_thrust_allocator(cudaStream_t stream = cudaStreamDefault) {
    return std::make_unique<ThrustAllocator<T>>(get_mm(), stream);
}

template<typename T>
void allocate_mem(T** ptr, bool clear = true, size_t sz = sizeof(T), cudaStream_t stream = cudaStreamDefault) {
    assert(ptr != nullptr);
    get_mm()->allocate((void**)ptr, clear, sz, stream);
}

template<typename T>
void release_mem(T* ptr, cudaStream_t stream = cudaStreamDefault) {
    get_mm()->release(ptr, stream);
}

void release_all_mem();
void debug_device_mem_usage();