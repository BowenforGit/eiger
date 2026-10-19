// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "memory/column.hpp"

namespace gamax::benchmark::concurrency {

inline constexpr char k_expr_name[] = "expr.two_col_add.nofuse";
inline constexpr char k_gather_name[] = "misc.gather.thrust";
inline constexpr char k_multiset_insert_name[] = "containers.multiset.insert";
inline constexpr char k_multiset_retrieve_name[] = "containers.multiset.retrieve";
inline constexpr int k_data_seed = 42;

[[noreturn]] inline void fail(std::string const& message)
{
    throw std::runtime_error(message);
}

inline void cuda_check(cudaError_t status, std::string_view what)
{
    if (status != cudaSuccess) {
        fail(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

struct Stream {
    cudaStream_t value{};

    explicit Stream(unsigned flags = cudaStreamNonBlocking)
    {
        cuda_check(cudaStreamCreateWithFlags(&value, flags), "cudaStreamCreateWithFlags");
    }
    ~Stream() { if (value) { cudaStreamDestroy(value); } }
    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;
};

struct Event {
    cudaEvent_t value{};

    Event() { cuda_check(cudaEventCreate(&value), "cudaEventCreate"); }
    ~Event() { if (value) { cudaEventDestroy(value); } }
    Event(Event const&) = delete;
    Event& operator=(Event const&) = delete;
};

struct GenericConfig {
    int size_exponent{};
    std::uint64_t size() const { return std::uint64_t{1} << size_exponent; }
};

inline std::uint64_t effective_chunk_size(std::uint64_t size, std::uint64_t requested)
{
    if (requested == 0) { fail("chunk size must be positive"); }
    return std::min(size, requested);
}

inline std::uint64_t chunk_count_for(std::uint64_t size, std::uint64_t chunk_size)
{
    return (size + chunk_size - 1) / chunk_size;
}

class PreparedOperation {
public:
    virtual ~PreparedOperation() = default;
    virtual void reset(cudaStream_t) {}
    virtual gamax::column_ptr enqueue(cudaStream_t stream) = 0;

    // begin_chunked preserves each operation's existing allocation boundary:
    // output-producing operations allocate one full output here, inside timing.
    virtual void begin_chunked(std::uint64_t chunk_size, cudaStream_t stream) = 0;
    virtual std::uint64_t chunk_count() const = 0;
    virtual void enqueue_chunk(std::uint64_t chunk_index, cudaStream_t stream) = 0;
    virtual gamax::column_ptr finish_chunked() = 0;

    gamax::column_ptr enqueue_chunked(std::uint64_t chunk_size, cudaStream_t stream)
    {
        begin_chunked(chunk_size, stream);
        for (std::uint64_t chunk = 0; chunk < chunk_count(); ++chunk) {
            enqueue_chunk(chunk, stream);
        }
        return finish_chunked();
    }

    virtual void retire_completed() {}
    virtual void validate(gamax::column_ptr const& result, cudaStream_t stream) const = 0;
};

class OperationDescriptor {
public:
    virtual ~OperationDescriptor() = default;
    virtual std::string stable_name() const = 0;
    virtual std::string axis_name() const = 0;
    virtual std::string config_string(GenericConfig const& config) const = 0;
    virtual void validate_config(GenericConfig const& config) const = 0;
    virtual std::size_t estimate_memory(GenericConfig const& config) const = 0;
    virtual std::unique_ptr<PreparedOperation> prepare(
        GenericConfig const& config, cudaStream_t setup_stream) const = 0;
};

class OperationRegistry {
public:
    OperationRegistry();
    OperationDescriptor const& lookup(std::string const& name) const;
    void list(std::ostream& out) const;

private:
    void add(std::unique_ptr<OperationDescriptor> descriptor);
    std::map<std::string, std::unique_ptr<OperationDescriptor>> operations_;
};

} // namespace gamax::benchmark::concurrency
