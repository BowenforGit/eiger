// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include "memory/mem_manager.hpp"

#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cassert>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace gamax {

class device_buffer {
public:
    // Project-empty buffers intentionally do not construct an RMM buffer or
    // capture a resource generation. They are immutable except by move.
    device_buffer() = default;

    device_buffer(
        std::size_t size,
        cudaStream_t stream,
        buffer_init init = buffer_init::uninitialized,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
        : size_{size}, alignment_{alignment}, empty_stream_{stream}
    {
        if (size == 0) {
            device_ = checked_current_device();
            return;
        }
        // Every allocation boundary validates the thread-local CUDA device;
        // TLS only caches the already validated resource generation.
        auto resource = resource_.acquire_for_buffer(device_);
        buffer_.emplace(
            size, alignment, rmm::cuda_stream_view{stream}, std::move(resource));
        if (init == buffer_init::zero) {
            auto const error = cudaMemsetAsync(buffer_->data(), 0, size, stream);
            if (error != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMemsetAsync failed: ") + cudaGetErrorString(error));
            }
        }
    }

    device_buffer(device_buffer const&) = delete;
    device_buffer& operator=(device_buffer const&) = delete;

    device_buffer(device_buffer&& other) noexcept
        : resource_{std::move(other.resource_)}
        , buffer_{std::move(other.buffer_)}
        , size_{std::exchange(other.size_, 0)}
        , alignment_{std::exchange(other.alignment_, rmm::CUDA_ALLOCATION_ALIGNMENT)}
        , device_{std::exchange(other.device_, -1)}
        , empty_stream_{std::exchange(other.empty_stream_, cudaStream_t{})}
    {
    }

    device_buffer& operator=(device_buffer&& other) noexcept
    {
        if (this == &other) { return *this; }
        assert_current_device();
        // Destroy the old RMM buffer before releasing its generation lease.
        buffer_.reset();
        resource_ = std::move(other.resource_);
        buffer_ = std::move(other.buffer_);
        size_ = std::exchange(other.size_, 0);
        alignment_ = std::exchange(other.alignment_, rmm::CUDA_ALLOCATION_ALIGNMENT);
        device_ = std::exchange(other.device_, -1);
        empty_stream_ = std::exchange(other.empty_stream_, cudaStream_t{});
        return *this;
    }

    ~device_buffer() noexcept { assert_current_device(); }

    [[nodiscard]] void* data() noexcept { return buffer_ ? buffer_->data() : nullptr; }
    [[nodiscard]] void const* data() const noexcept { return buffer_ ? buffer_->data() : nullptr; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] int device_id() const noexcept { return device_; }

    [[nodiscard]] cudaStream_t stream() const noexcept
    {
        return buffer_ ? buffer_->stream().value() : empty_stream_;
    }

    // Sets only the stream used by future deallocation. It does not record an
    // event, migrate work, or make prior uses on another stream safe.
    void set_stream(cudaStream_t stream) noexcept
    {
        empty_stream_ = stream;
        if (buffer_) { buffer_->set_stream(rmm::cuda_stream_view{stream}); }
    }

private:
    static int current_device() noexcept
    {
        int device = -1;
        auto const result = cudaGetDevice(&device);
        assert(result == cudaSuccess);
        return result == cudaSuccess ? device : -1;
    }

    static int checked_current_device()
    {
        int device = -1;
        auto const result = cudaGetDevice(&device);
        if (result != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDevice failed: ") + cudaGetErrorString(result));
        }
        return device;
    }

    void assert_current_device() const noexcept
    {
#ifndef NDEBUG
        if (buffer_) { assert(current_device() == device_); }
#endif
    }

    // RMM object must be destroyed before the resource handle/lease.
    resource_handle resource_{};
    std::optional<rmm::device_buffer> buffer_{};
    std::size_t size_{};
    std::size_t alignment_{rmm::CUDA_ALLOCATION_ALIGNMENT};
    int device_{-1};
    cudaStream_t empty_stream_{};
};

} // namespace gamax
