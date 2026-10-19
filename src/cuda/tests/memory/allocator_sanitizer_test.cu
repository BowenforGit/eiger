// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <gtest/gtest.h>

#include "memory/device_buffer.hpp"
#include "memory/mem_manager.hpp"

#include <array>
#include <stdexcept>

namespace {

gamax::memory_resource_options direct_tracked_options()
{
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::cuda;
    options.tracking = gamax::memory_tracking_mode::on;
    return options;
}

TEST(AllocatorSanitizer, AllocationAndFirstUseOnSameStreamAreOrdered)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    std::array<unsigned char, 256> host{};
    {
        gamax::device_buffer buffer{host.size(), stream};
        ASSERT_EQ(cudaMemsetAsync(buffer.data(), 0x5a, buffer.size(), stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
            host.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }
    for (auto value : host) { EXPECT_EQ(value, 0x5a); }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorSanitizer, EventHandoffOrdersFirstUseOnAnotherStream)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t allocation_stream{}, use_stream{};
    cudaEvent_t ready{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&allocation_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&use_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), cudaSuccess);
    std::array<unsigned char, 512> host{};
    {
        gamax::device_buffer buffer{host.size(), allocation_stream};
        ASSERT_EQ(cudaMemsetAsync(
            buffer.data(), 0x33, buffer.size(), allocation_stream), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(ready, allocation_stream), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(use_stream, ready), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
            host.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost, use_stream),
            cudaSuccess);
        buffer.set_stream(use_stream);
        ASSERT_EQ(cudaStreamSynchronize(use_stream), cudaSuccess);
    }
    for (auto value : host) { EXPECT_EQ(value, 0x33); }
    ASSERT_EQ(cudaEventDestroy(ready), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(use_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(allocation_stream), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorSanitizer, MultiConsumerEventsJoinBeforeRetirement)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t first{}, second{}, retirement{};
    cudaEvent_t first_done{}, second_done{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&retirement, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&first_done, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&second_done, cudaEventDisableTiming), cudaSuccess);
    std::array<unsigned char, 1'024> host{};
    {
        gamax::device_buffer buffer{host.size(), first};
        auto* bytes = static_cast<unsigned char*>(buffer.data());
        ASSERT_EQ(cudaMemsetAsync(bytes, 0x11, host.size() / 2, first), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(
            bytes + host.size() / 2, 0x22, host.size() / 2, second), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(first_done, first), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(second_done, second), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(retirement, first_done), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(retirement, second_done), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
            host.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost, retirement),
            cudaSuccess);
        buffer.set_stream(retirement);
    }
    ASSERT_EQ(cudaStreamSynchronize(retirement), cudaSuccess);
    for (std::size_t index = 0; index < host.size(); ++index) {
        EXPECT_EQ(host[index], index < host.size() / 2 ? 0x11 : 0x22);
    }
    ASSERT_EQ(cudaEventDestroy(second_done), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(first_done), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(retirement), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(second), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(first), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorSanitizer, ExceptionAfterSubmissionJoinsStreamsBeforeOwnersUnwind)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());

    cudaStream_t first{}, second{}, retirement{};
    cudaEvent_t ready{}, first_done{}, second_done{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&retirement, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&first_done, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&second_done, cudaEventDisableTiming), cudaSuccess);

    std::array<unsigned char, 1'024> host{};
    EXPECT_THROW(
        ([&] {
            gamax::device_buffer owner{host.size(), first};
            ASSERT_EQ(cudaEventRecord(ready, first), cudaSuccess);
            ASSERT_EQ(cudaStreamWaitEvent(second, ready), cudaSuccess);
            try {
                auto* bytes = static_cast<unsigned char*>(owner.data());
                ASSERT_EQ(cudaMemsetAsync(bytes, 0x11, host.size() / 2, first), cudaSuccess);
                ASSERT_EQ(cudaMemsetAsync(
                    bytes + host.size() / 2, 0x22, host.size() / 2, second),
                    cudaSuccess);
                throw std::runtime_error("injected failure after GPU submission");
            } catch (...) {
                ASSERT_EQ(cudaEventRecord(first_done, first), cudaSuccess);
                ASSERT_EQ(cudaEventRecord(second_done, second), cudaSuccess);
                ASSERT_EQ(cudaStreamWaitEvent(retirement, first_done), cudaSuccess);
                ASSERT_EQ(cudaStreamWaitEvent(retirement, second_done), cudaSuccess);
                ASSERT_EQ(cudaMemcpyAsync(
                    host.data(), owner.data(), owner.size(),
                    cudaMemcpyDeviceToHost, retirement), cudaSuccess);
                owner.set_stream(retirement);
                throw;
            }
        }()),
        std::runtime_error);

    ASSERT_EQ(cudaStreamSynchronize(retirement), cudaSuccess);
    for (std::size_t index = 0; index < host.size(); ++index) {
        EXPECT_EQ(host[index], index < host.size() / 2 ? 0x11 : 0x22);
    }
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);

    ASSERT_EQ(cudaEventDestroy(second_done), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(first_done), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(ready), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(retirement), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(second), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(first), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

} // namespace
