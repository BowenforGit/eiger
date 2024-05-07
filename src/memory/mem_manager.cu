#include "mem_manager.hpp"

#include <unordered_map>
#include <iostream>
#include <cuda.h>
#include <list>
#include <stdexcept>
#include <memory>

#include "utils/check_error.hpp"

std::shared_ptr<UserSpaceMM> mm {nullptr};

std::shared_ptr<UserSpaceMM> get_mm() {
    if(mm.get() == nullptr) {
        mm = std::make_shared<UserSpaceMM>(mem_pool_size);
    }
    return mm;
}

void delete_mm() {
    mm.reset();
}

void debug_cuda_mem_usage() {
#ifdef USE_CUDA_MEMALLOC
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    std::cout << "Free bytes : " << free << " Total bytes : " << total << std::endl;
#else
    auto mm = get_mm();
    std::cout << "Peak memory usage: " << mm->get_peak_mem_used() << " bytes (" << mm->get_peak_mem_used() / 1024 / 1024 / 1024 << " GB)\n";
    std::cout << "Cuurent memory usage: " << mm->get_mem_used() << " bytes (" << mm->get_mem_used() / 1024 / 1024 / 1024 << " GB)\n";
#endif
}

void release_all_mem() {
#ifdef USE_CUDA_MEMALLOC
    std::cout << "Cannot release all memory with the CUDA memory allocator\n";
#else
    delete_mm();
#endif
}

namespace gamax::details {
    void alloc_by_cuda(void** ptr, bool clear, size_t sz, cudaStream_t stream) {
        CHECK_CUDA_ERROR(cudaMallocAsync(ptr, sz, stream));
        if(clear) CHECK_CUDA_ERROR(cudaMemsetAsync(*ptr, 0, sz, stream));
    }

    size_t get_cuda_free_mem() {
        size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        std::cout << free << std::endl;

        return free;
    }

    void alloc_by_rmm_mempool(void** ptr, bool clear, size_t sz, cudaStream_t stream) {
        get_mm()->allocate(ptr, clear, sz, stream);
    }

    void free_rmm_mempool(void* ptr, cudaStream_t stream) {
        get_mm()->release(ptr, stream);
    }
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

size_t UserSpaceMM::get_peak_mem_used() const {
    return peak_mem_used;
}

size_t UserSpaceMM::get_mem_used() const {
    return mem_used;
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

    int i = 0;
    for(auto b = free_list.begin(); b != free_list.end(); b++, i++) {
        std::cout << "Block " << i << ": " << b->sz << " bytes\n";
    }
    std::cout << "Peak memory usage: " << peak_mem_used << " bytes (" << peak_mem_used / 1024 / 1024 / 1024 << " GB)\n";
    std::cout << "Cuurent memory usage: " << mem_used << " bytes (" << mem_used / 1024 / 1024 / 1024 << " GB)\n";
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