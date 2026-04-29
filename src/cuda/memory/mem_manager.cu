#include "mem_manager.hpp"

#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <cuda.h>
#include <list>
#include <stdexcept>
#include <memory>

#include "utils/check_error.hpp"

std::shared_ptr<MemoryManager> mm {nullptr};

namespace {
    // The kind to instantiate on the next get_mm() call. Default ==
    // UserSpaceMM (production behaviour). use_cuda_mm() flips it to
    // CudaMM. Sticky across release_all_mem() / re-init cycles, so
    // a single call at startup affects every later cell.
    enum class MMKind { UserSpace, Cuda };
    MMKind requested_kind = MMKind::UserSpace;
}

void use_cuda_mm() {
    if (mm) {
        throw std::runtime_error(
            "use_cuda_mm(): the global memory manager is already "
            "initialized; call this before any allocation");
    }
    requested_kind = MMKind::Cuda;
}

std::shared_ptr<MemoryManager> get_mm() {
    if(!mm) {
        if (requested_kind == MMKind::Cuda) {
            mm = std::make_shared<CudaMM>();
        } else {
            // Force CUDA context init before sizing the slab.
            // cudaMemGetInfo returns (0, 0) on a cold context (no
            // prior CUDA call), which would size the pool to 0 and
            // make cudaMalloc(0) return a null pointer — surfaced as
            // "Initializing memory pool fails" in the probe binary
            // when called before the resolver's first cudaMallocHost.
            // Force CUDA context init before sizing the slab.
            // cudaMemGetInfo returns (0, 0) on a cold context (the
            // CUDA driver lazy-inits the device context only on the
            // first device-affecting alloc). cudaFree(nullptr),
            // cudaSetDevice, cudaDeviceSynchronize are all no-ops on
            // a cold context. A 4-byte cudaMalloc + cudaFree pair
            // reliably forces context init; then cudaGetLastError
            // clears the swallowed status.
            void* ctx_init = nullptr;
            cudaMalloc(&ctx_init, 4);
            cudaFree(ctx_init);
            cudaGetLastError();
            std::size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            // free - 128 MiB leaves a fixed scratch margin for the
            // CUDA driver, NCCL, etc. Earlier 0.95×free squeezed the
            // slab by ~4 GiB of unnecessary headroom; 256 MiB still
            // left Q1 SF=100 ~kB short of fitting after Project
            // aliasing. 128 MiB is enough scratch in practice
            // (cudaMalloc itself uses much less) and gives the slab
            // another 128 MiB of working space.
            constexpr std::size_t kDriverReserve = std::size_t{128} << 20;
            auto mem_pool_size = (free > kDriverReserve)
                                 ? (free - kDriverReserve) : free;
            mm = std::make_shared<UserSpaceMM>(mem_pool_size);
        }
    }
    return mm;
}

void debug_device_mem_usage() {
    std::cout << "Peak memory usage: " << mm->get_peak_mem_used() << " bytes (" << mm->get_peak_mem_used() / 1024 / 1024 / 1024 << " GB)\n";
    std::cout << "Cuurent memory usage: " << mm->get_mem_used() << " bytes (" << mm->get_mem_used() / 1024 / 1024 / 1024 << " GB)\n";
}

void release_all_mem() {
    // Drop the manager — UserSpaceMM's dtor cudaFree's its slab;
    // CudaMM's alloc_map should already be empty by the time callers
    // reach this point (callers release intermediates via column
    // dtors first). The next get_mm() rebuilds per `requested_kind`.
    mm.reset();
}

UserSpaceMM::UserSpaceMM(const size_t kInitialPoolSize, const size_t kAlignBits) : align_bits(kAlignBits) {
    std::cout << "Memory pool has initial size " << kInitialPoolSize << " bytes.\n";
    cudaMalloc(&root, kInitialPoolSize);
    if(!root) {
        throw std::runtime_error("Initializing memory pool fails.\n");
    }

    block_type root_block = {.data = (char*)root, .sz = kInitialPoolSize};
    free_list.insert(free_list.cend(), root_block);
}

UserSpaceMM::~UserSpaceMM() {
    if(root) cudaFree(root);
}

void UserSpaceMM::allocate(void** ptr, bool clear, size_t sz, cudaStream_t stream) {
    if(sz == 0) {
        std::cout << "!!WARNING!! Allocation of 0 bytes!!!\n";
        *ptr = nullptr;
        return;
    }

    auto b = find_block(sz);
    *ptr = (void*)(roundup(b.data));
    if(clear) cudaMemsetAsync(*ptr, 0, sz, stream);

    alloc_map[*ptr] = b;
    mem_used += b.sz;
    if(mem_used > peak_mem_used) peak_mem_used = mem_used;
}

void UserSpaceMM::release(void* ptr, cudaStream_t stream) {
    if(auto search = alloc_map.find(ptr); search != alloc_map.end()) {
        insert_block(search->second);
        alloc_map.erase(search);
        mem_used -= search->second.sz;
    }
    else {
        std::cout << "!!WARNING!! Potential double-free: " 
                    << ptr << " is not allocated through MemoryManager.\n";
    }
}

// find a block in the free list that is at least as large as sz
UserSpaceMM::block_type UserSpaceMM::find_block(const size_t sz) {
    block_type ret;
    
    for(auto b = free_list.begin(); b != free_list.end(); b++) {
        if(can_alloc(*b, sz)) {
            char* alloc_start = roundup(b->data);
            
            ret.data = b->data;
            ret.sz = sz + alloc_start - b->data;

            if(b->sz == ret.sz) {
                free_list.erase(b);
            } else {
                b->data += ret.sz;
                b->sz -= ret.sz;
            }

            return ret;
        }
    }

    size_t i = 0;
    size_t total_free = 0, largest_free = 0;
    for(auto b = free_list.begin(); b != free_list.end(); b++, i++) {
        std::cout << "Block " << i << ": " << b->sz << " bytes\n";
        total_free += b->sz;
        if (b->sz > largest_free) largest_free = b->sz;
    }
    std::cout << "Peak memory usage: " << peak_mem_used << " bytes (" << peak_mem_used / 1024 / 1024 / 1024 << " GB)\n";
    std::cout << "Cuurent memory usage: " << mem_used << " bytes (" << mem_used / 1024 / 1024 / 1024 << " GB)\n";
    std::cout << "Failed alloc: requested=" << sz << " bytes ("
              << (sz + 1024*1024 - 1) / (1024*1024) << " MiB)"
              << ", total_free=" << total_free
              << " (" << total_free / (1024*1024) << " MiB)"
              << ", largest_free=" << largest_free
              << " (" << largest_free / (1024*1024) << " MiB)\n";
    throw std::bad_alloc();
}

// Insert the block back to the free list, merge if necessary
void UserSpaceMM::insert_block(const block_type& free) {
    if(free_list.empty()) {
        free_list.insert(free_list.cend(), free);
        return;
    }

    auto const next = std::find_if(free_list.begin(), free_list.end(), [free](block_type const& blk) { return free.data < blk.data; });
    auto const prev = (next == free_list.cbegin()) ? next : std::prev(next);

    // Coalesce with neighboring blocks or insert the new block if it can't be coalesced
    bool const merge_prev = (prev->data + prev->sz == free.data);
    bool const merge_next = (next != free_list.cend()) && (free.data + free.sz == next->data);

    if (merge_prev && merge_next) {
        prev->sz += free.sz + next->sz;
        free_list.erase(next);
    } else if (merge_prev) {
        prev->sz += free.sz;
    } else if (merge_next) {
        next->data = free.data;
        next->sz += free.sz;
    } else {
        free_list.insert(next, free);  // cannot be coalesced, just insert
    }
}

size_t UserSpaceMM::roundup(const size_t value) {
    return (value + (align_bits - 1)) & ~(align_bits - 1);
}

char* UserSpaceMM::roundup(const char* ptr) {
    uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    value = (value + (align_bits - 1)) & ~(align_bits - 1);
    return (char*)value;
}

bool UserSpaceMM::can_alloc(const block_type& block, const size_t sz) {
    auto ptr = roundup(block.data);
    return ptr + sz < block.data + block.sz;
}

void CudaMM::allocate(void** ptr, bool clear, size_t sz, cudaStream_t stream)
{
    CHECK_CUDA_ERROR(cudaMallocAsync(ptr, sz, stream));
    if(clear) CHECK_CUDA_ERROR(cudaMemsetAsync(*ptr, 0, sz, stream));

    mem_used += sz;
    peak_mem_used = std::max(peak_mem_used, mem_used);
    alloc_map[*ptr] = sz;
}

void CudaMM::release(void* ptr, cudaStream_t stream)
{
    auto found = alloc_map.find(ptr);
    if(found == alloc_map.end()) {
        throw std::runtime_error("Freeing a non-allocated pointer in CUDA Memory Manager.");
    }
    mem_used -= found->second;
    CHECK_CUDA_ERROR(cudaFreeAsync(ptr, stream));
    alloc_map.erase(found);
}
