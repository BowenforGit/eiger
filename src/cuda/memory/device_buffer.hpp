#pragma once

#include <iostream>
#include <cuda_runtime_api.h>
#include <utility>
#include "memory/mem_manager.hpp"

namespace gamax {
class device_buffer {
public:
    __host__ __device__ device_buffer(void* data, size_t size) : data_(data), size_(size) {}
    __host__ device_buffer(size_t size, bool clear = false) : size_(size) {
        allocate_mem(&data_, clear, size_);
    }
    
    // not copy-constructible or copy-assignable
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    // move constructor and move assignment
    __host__ __device__ device_buffer(device_buffer&& other) noexcept 
        : data_(other.data_)
        , size_(other.size_) {
            /* We avoid using std::exchange because it does not work for the device code */
            other.data_ = nullptr;
            other.size_ = 0;
        }
    
    __host__ __device__ device_buffer& operator=(device_buffer&& other) noexcept {
        if (this != &other) {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
            release_mem(data_);
#endif
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    __host__ __device__ ~device_buffer() {
        if(data_ == nullptr) return;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
        release_mem(data_);
#endif
    }

    __host__ __device__ void* data() const { return data_; }
    __host__ __device__ size_t size() const { return size_; }

private:
    void* data_ {nullptr};
    size_t size_ {0}; // in bytes
};
} // namespace gamax
