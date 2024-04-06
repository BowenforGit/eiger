#pragma once

#include <iostream>
#include "utils/utils.cuh"

namespace gamax {
class device_buffer {
public:
    device_buffer(void* data, size_t size) : data_(data), size_(size) {}
    device_buffer(size_t size, bool clear = false) : size_(size) {
        allocate_mem(&data_, clear, size_);
    }
    
    // not copy-constructible or copy-assignable
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    // move constructor and move assignment
    device_buffer(device_buffer&& other) noexcept 
        : data_(std::exchange(other.data_, nullptr))
        , size_(std::exchange(other.size_, 0)) {}
    
    device_buffer& operator=(device_buffer&& other) noexcept {
        if (this != &other) {
            release_mem(data_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~device_buffer() {
        if(data_ == nullptr) return;
        release_mem(data_);
    }

    void* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void* data_ {nullptr};
    size_t size_ {0}; // in bytes
};
} // namespace gamax