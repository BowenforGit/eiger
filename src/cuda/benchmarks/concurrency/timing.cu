// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "timing.hpp"

#include <cuda/std/chrono>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>

#include "memory/device_buffer.hpp"

namespace gamax::benchmark::concurrency {
namespace {

__global__ void blocking_gate_kernel(
    volatile int const* release, volatile int* timed_out, unsigned long long timeout_cycles)
{
    unsigned long long const start = clock64();
    while (*release == 0 && clock64() - start < timeout_cycles) { __nanosleep(1'000); }
    if (*release == 0) {
        *timed_out = 1;
        __threadfence_system();
    }
}

__global__ void spin_kernel(unsigned long long cycles)
{
    unsigned long long const start = clock64();
    while (clock64() - start < cycles) { __nanosleep(1'000); }
}

class SimultaneousGate {
public:
    SimultaneousGate(int clock_rate_khz, double timeout_seconds)
        : timeout_cycles_{static_cast<unsigned long long>(
              timeout_seconds * static_cast<double>(clock_rate_khz) * 1000.0)}
    {
        cuda_check(cudaHostAlloc(
            reinterpret_cast<void**>(&host_release_), sizeof(int), cudaHostAllocMapped),
            "cudaHostAlloc(gate release)");
        try {
            cuda_check(cudaHostAlloc(
                reinterpret_cast<void**>(&host_timeout_), sizeof(int), cudaHostAllocMapped),
                "cudaHostAlloc(gate timeout)");
            cuda_check(cudaHostGetDevicePointer(&device_release_, host_release_, 0),
                       "cudaHostGetDevicePointer(gate release)");
            cuda_check(cudaHostGetDevicePointer(&device_timeout_, host_timeout_, 0),
                       "cudaHostGetDevicePointer(gate timeout)");
            Stream warmup;
            *host_release_ = 1;
            *host_timeout_ = 0;
            blocking_gate_kernel<<<1, 1, 0, warmup.value>>>(
                device_release_, device_timeout_, timeout_cycles_);
            cuda_check(cudaGetLastError(), "blocking gate warmup launch");
            cuda_check(cudaStreamSynchronize(warmup.value), "blocking gate warmup");
        } catch (...) {
            cudaFreeHost(host_timeout_);
            cudaFreeHost(host_release_);
            host_timeout_ = nullptr;
            host_release_ = nullptr;
            throw;
        }
    }

    ~SimultaneousGate()
    {
        if (host_timeout_) { cudaFreeHost(host_timeout_); }
        if (host_release_) { cudaFreeHost(host_release_); }
    }

    void begin()
    {
        *host_release_ = 0;
        *host_timeout_ = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void block(cudaStream_t stream)
    {
        blocking_gate_kernel<<<1, 1, 0, stream>>>(
            device_release_, device_timeout_, timeout_cycles_);
        cuda_check(cudaGetLastError(), "blocking gate launch");
    }

    void release()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *host_release_ = 1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void check(std::string_view context = {}) const
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (*host_timeout_) {
            fail("simultaneous-start gate timed out" +
                 (context.empty() ? std::string{} : " while " + std::string(context)) +
                 "; an operation may synchronize in enqueue()");
        }
    }

private:
    int* host_release_{};
    int* host_timeout_{};
    int* device_release_{};
    int* device_timeout_{};
    unsigned long long timeout_cycles_{};
};

double elapsed(cudaEvent_t start, cudaEvent_t stop)
{
    float value = 0.0f;
    cuda_check(cudaEventElapsedTime(&value, start, stop), "cudaEventElapsedTime");
    return value;
}

std::uint32_t sample_color(SampleSpec sample)
{
    if (sample.variant == ExecutionVariant::chunked) {
        switch (sample.mode) {
            case Mode::a_alone: return 0xff76a5d5;
            case Mode::b_alone: return 0xffffbe69;
            case Mode::concurrent: return 0xffd3a6cc;
            default: return 0xffbab0ac;
        }
    }
    switch (sample.mode) {
        case Mode::a_alone: return 0xff4e79a7;
        case Mode::b_alone: return 0xfff28e2b;
        case Mode::sequential_ab: return 0xff59a14f;
        case Mode::sequential_ba: return 0xffe15759;
        case Mode::concurrent: return 0xffb07aa1;
    }
    return 0xffbab0ac;
}

} // namespace

char const* mode_name(Mode mode)
{
    switch (mode) {
        case Mode::a_alone: return "a_isolated";
        case Mode::b_alone: return "b_isolated";
        case Mode::sequential_ab: return "sequential_a_then_b";
        case Mode::sequential_ba: return "sequential_b_then_a";
        case Mode::concurrent: return "concurrent";
    }
    return "unknown";
}

char const* variant_name(ExecutionVariant variant)
{
    return variant == ExecutionVariant::whole ? "whole" : "chunked";
}

std::string sample_name(SampleSpec sample)
{
    if (sample.variant == ExecutionVariant::whole) { return mode_name(sample.mode); }
    return std::string("chunked/") + mode_name(sample.mode);
}

std::vector<SampleSpec> all_samples(bool include_chunked)
{
    std::vector<SampleSpec> samples{
        {Mode::a_alone, ExecutionVariant::whole},
        {Mode::b_alone, ExecutionVariant::whole},
        {Mode::sequential_ab, ExecutionVariant::whole},
        {Mode::sequential_ba, ExecutionVariant::whole},
        {Mode::concurrent, ExecutionVariant::whole}};
    if (include_chunked) {
        samples.push_back({Mode::a_alone, ExecutionVariant::chunked});
        samples.push_back({Mode::b_alone, ExecutionVariant::chunked});
        samples.push_back({Mode::concurrent, ExecutionVariant::chunked});
    }
    return samples;
}

ScopedSampleNvtxRange::ScopedSampleNvtxRange(SampleSpec sample)
{
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = sample_color(sample);
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    auto const name = sample_name(sample);
    attributes.message.ascii = name.c_str();
    nvtxRangePushEx(&attributes);
}

ScopedSampleNvtxRange::~ScopedSampleNvtxRange() { nvtxRangePop(); }

class TimingCore::Impl {
public:
    Impl(int device_clock_khz, double gate_timeout_seconds)
        : gate_{device_clock_khz, gate_timeout_seconds} {}

    std::pair<gamax::column_ptr, gamax::column_ptr> prime_whole(
        PreparedOperation& a, PreparedOperation& b)
    {
        a.reset(stream_a_.value);
        b.reset(stream_b_.value);
        cuda_check(cudaStreamSynchronize(stream_a_.value), "prime A reset on stream A");
        cuda_check(cudaStreamSynchronize(stream_b_.value), "prime B reset on stream B");
        auto a_on_a = a.enqueue(stream_a_.value);
        auto b_on_b = b.enqueue(stream_b_.value);
        cuda_check(cudaStreamSynchronize(stream_a_.value), "prime A on stream A");
        cuda_check(cudaStreamSynchronize(stream_b_.value), "prime B on stream B");
        a.retire_completed();
        b.retire_completed();

        b.reset(stream_a_.value);
        a.reset(stream_b_.value);
        cuda_check(cudaStreamSynchronize(stream_a_.value), "prime B reset on stream A");
        cuda_check(cudaStreamSynchronize(stream_b_.value), "prime A reset on stream B");
        auto b_on_a = b.enqueue(stream_a_.value);
        auto a_on_b = a.enqueue(stream_b_.value);
        cuda_check(cudaStreamSynchronize(stream_a_.value), "prime B on stream A");
        cuda_check(cudaStreamSynchronize(stream_b_.value), "prime A on stream B");
        a.retire_completed();
        b.retire_completed();
        b_on_a.reset();
        a_on_b.reset();
        cuda_check(cudaStreamSynchronize(stream_a_.value), "prime stream A retirement");
        cuda_check(cudaStreamSynchronize(stream_b_.value), "prime stream B retirement");
        return {std::move(a_on_a), std::move(b_on_b)};
    }

    Timing run(
        SampleSpec sample, PreparedOperation& a, PreparedOperation& b,
        bool a_first, std::uint64_t a_chunk_size, std::uint64_t b_chunk_size)
    {
        a.retire_completed();
        b.retire_completed();
        if (sample.variant == ExecutionVariant::chunked &&
            sample.mode != Mode::a_alone && sample.mode != Mode::b_alone &&
            sample.mode != Mode::concurrent) {
            fail("chunked sequential sampling is not supported");
        }
        switch (sample.mode) {
            case Mode::a_alone:
                return isolated(a, stream_a_.value, true, sample.variant, a_chunk_size);
            case Mode::b_alone:
                return isolated(b, stream_b_.value, false, sample.variant, b_chunk_size);
            case Mode::sequential_ab: return sequential(a, b, true);
            case Mode::sequential_ba: return sequential(a, b, false);
            case Mode::concurrent:
                return concurrent(a, b, a_first, sample.variant, a_chunk_size, b_chunk_size);
        }
        fail("unreachable timing mode");
    }

private:
    Timing isolated(
        PreparedOperation& operation, cudaStream_t stream, bool is_a,
        ExecutionVariant variant, std::uint64_t chunk_size)
    {
        Event start, stop;
        gate_.begin();
        gate_.block(stream);
        cuda_check(cudaEventRecord(start.value, stream), "isolated start record");
        gamax::column_ptr result;
        if (variant == ExecutionVariant::whole) {
            result = operation.enqueue(stream);
        } else {
            operation.begin_chunked(chunk_size, stream);
            for (std::uint64_t chunk = 0; chunk < operation.chunk_count(); ++chunk) {
                operation.enqueue_chunk(chunk, stream);
                gate_.check("enqueuing isolated chunks");
            }
            result = operation.finish_chunked();
        }
        gate_.check(is_a ? "enqueuing operation A" : "enqueuing operation B");
        cuda_check(cudaEventRecord(stop.value, stream), "isolated stop record");
        gate_.release();
        cuda_check(cudaEventSynchronize(stop.value), "isolated stop synchronization");
        gate_.check();

        Timing timing;
        timing.a_ms = timing.b_ms = std::numeric_limits<double>::quiet_NaN();
        timing.start_skew_ms = std::numeric_limits<double>::quiet_NaN();
        double const duration = elapsed(start.value, stop.value);
        timing.order = is_a ? "a" : "b";
        timing.makespan_ms = duration;
        if (is_a) {
            timing.a_ms = duration;
            timing.a_result = std::move(result);
        } else {
            timing.b_ms = duration;
            timing.b_result = std::move(result);
        }
        return timing;
    }

    Timing sequential(PreparedOperation& a, PreparedOperation& b, bool a_then_b)
    {
        Event start_first, stop_first, start_second, stop_second;
        cudaStream_t first_stream = a_then_b ? stream_a_.value : stream_b_.value;
        cudaStream_t second_stream = a_then_b ? stream_b_.value : stream_a_.value;
        PreparedOperation& first = a_then_b ? a : b;
        PreparedOperation& second = a_then_b ? b : a;

        gate_.begin();
        gate_.block(first_stream);
        cuda_check(cudaEventRecord(start_first.value, first_stream), "sequential first start");
        auto first_result = first.enqueue(first_stream);
        gate_.check(a_then_b ? "enqueuing sequential A" : "enqueuing sequential B");
        cuda_check(cudaEventRecord(stop_first.value, first_stream), "sequential first stop");
        cuda_check(cudaStreamWaitEvent(second_stream, stop_first.value, 0),
                   "sequential dependency wait");
        cuda_check(cudaEventRecord(start_second.value, second_stream), "sequential second start");
        auto second_result = second.enqueue(second_stream);
        gate_.check(a_then_b ? "enqueuing sequential B" : "enqueuing sequential A");
        cuda_check(cudaEventRecord(stop_second.value, second_stream), "sequential second stop");
        gate_.release();
        cuda_check(cudaEventSynchronize(stop_second.value), "sequential stop synchronization");
        gate_.check();

        Timing timing;
        timing.start_skew_ms = std::numeric_limits<double>::quiet_NaN();
        double const first_ms = elapsed(start_first.value, stop_first.value);
        double const second_ms = elapsed(start_second.value, stop_second.value);
        timing.makespan_ms = elapsed(start_first.value, stop_second.value);
        timing.order = a_then_b ? "a_then_b" : "b_then_a";
        if (a_then_b) {
            timing.a_ms = first_ms;
            timing.b_ms = second_ms;
            timing.a_result = std::move(first_result);
            timing.b_result = std::move(second_result);
        } else {
            timing.a_ms = second_ms;
            timing.b_ms = first_ms;
            timing.b_result = std::move(first_result);
            timing.a_result = std::move(second_result);
        }
        return timing;
    }

    Timing concurrent(
        PreparedOperation& a, PreparedOperation& b, bool a_first,
        ExecutionVariant variant, std::uint64_t a_chunk_size, std::uint64_t b_chunk_size)
    {
        if (variant == ExecutionVariant::whole) { return concurrent_whole(a, b, a_first); }
        return concurrent_chunked(a, b, a_first, a_chunk_size, b_chunk_size);
    }

    Timing concurrent_whole(PreparedOperation& a, PreparedOperation& b, bool a_first)
    {
        Event start_a, stop_a, start_b, stop_b;
        gate_.begin();
        gate_.block(stream_a_.value);
        gate_.block(stream_b_.value);
        gamax::column_ptr result_a, result_b;
        auto enqueue_a = [&] {
            cuda_check(cudaEventRecord(start_a.value, stream_a_.value), "concurrent A start");
            result_a = a.enqueue(stream_a_.value);
            gate_.check("enqueuing concurrent A");
            cuda_check(cudaEventRecord(stop_a.value, stream_a_.value), "concurrent A stop");
        };
        auto enqueue_b = [&] {
            cuda_check(cudaEventRecord(start_b.value, stream_b_.value), "concurrent B start");
            result_b = b.enqueue(stream_b_.value);
            gate_.check("enqueuing concurrent B");
            cuda_check(cudaEventRecord(stop_b.value, stream_b_.value), "concurrent B stop");
        };
        if (a_first) { enqueue_a(); enqueue_b(); }
        else { enqueue_b(); enqueue_a(); }
        gate_.release();
        cuda_check(cudaEventSynchronize(stop_a.value), "concurrent A stop synchronization");
        cuda_check(cudaEventSynchronize(stop_b.value), "concurrent B stop synchronization");
        gate_.check();
        return concurrent_result(start_a, stop_a, start_b, stop_b, a_first,
                                 std::move(result_a), std::move(result_b));
    }

    Timing concurrent_chunked(
        PreparedOperation& a, PreparedOperation& b, bool a_first,
        std::uint64_t a_chunk_size, std::uint64_t b_chunk_size)
    {
        Event start_a, stop_a, start_b, stop_b;
        gate_.begin();
        gate_.block(stream_a_.value);
        gate_.block(stream_b_.value);

        auto begin_a = [&] {
            cuda_check(cudaEventRecord(start_a.value, stream_a_.value),
                       "chunked concurrent A start");
            a.begin_chunked(a_chunk_size, stream_a_.value);
        };
        auto begin_b = [&] {
            cuda_check(cudaEventRecord(start_b.value, stream_b_.value),
                       "chunked concurrent B start");
            b.begin_chunked(b_chunk_size, stream_b_.value);
        };
        if (a_first) { begin_a(); begin_b(); }
        else { begin_b(); begin_a(); }

        auto const a_chunks = a.chunk_count();
        auto const b_chunks = b.chunk_count();
        auto const rounds = std::max(a_chunks, b_chunks);
        gamax::column_ptr result_a, result_b;
        bool a_stopped = false;
        bool b_stopped = false;
        auto enqueue_a = [&](std::uint64_t chunk) {
            if (chunk >= a_chunks) { return; }
            a.enqueue_chunk(chunk, stream_a_.value);
            if (chunk + 1 == a_chunks) {
                result_a = a.finish_chunked();
                cuda_check(cudaEventRecord(stop_a.value, stream_a_.value),
                           "chunked concurrent A stop");
                a_stopped = true;
            }
        };
        auto enqueue_b = [&](std::uint64_t chunk) {
            if (chunk >= b_chunks) { return; }
            b.enqueue_chunk(chunk, stream_b_.value);
            if (chunk + 1 == b_chunks) {
                result_b = b.finish_chunked();
                cuda_check(cudaEventRecord(stop_b.value, stream_b_.value),
                           "chunked concurrent B stop");
                b_stopped = true;
            }
        };
        for (std::uint64_t chunk = 0; chunk < rounds; ++chunk) {
            if (a_first) { enqueue_a(chunk); enqueue_b(chunk); }
            else { enqueue_b(chunk); enqueue_a(chunk); }
            gate_.check("enqueuing naturally interleaved chunks");
        }
        if (!a_stopped || !b_stopped) { fail("chunked operation produced zero chunks"); }

        gate_.release();
        cuda_check(cudaEventSynchronize(stop_a.value), "chunked A stop synchronization");
        cuda_check(cudaEventSynchronize(stop_b.value), "chunked B stop synchronization");
        gate_.check();
        return concurrent_result(start_a, stop_a, start_b, stop_b, a_first,
                                 std::move(result_a), std::move(result_b));
    }

    Timing concurrent_result(
        Event const& start_a, Event const& stop_a, Event const& start_b, Event const& stop_b,
        bool a_first, gamax::column_ptr result_a, gamax::column_ptr result_b)
    {
        Timing timing;
        timing.a_ms = elapsed(start_a.value, stop_a.value);
        timing.b_ms = elapsed(start_b.value, stop_b.value);
        timing.makespan_ms = std::max(timing.a_ms, timing.b_ms);
        float skew = 0.0f;
        cuda_check(cudaEventElapsedTime(&skew, start_a.value, start_b.value),
                   "concurrent start skew");
        timing.start_skew_ms = std::fabs(static_cast<double>(skew));
        timing.order = a_first ? "a_first" : "b_first";
        timing.a_result = std::move(result_a);
        timing.b_result = std::move(result_b);
        return timing;
    }

    Stream stream_a_;
    Stream stream_b_;
    SimultaneousGate gate_;
};

TimingCore::TimingCore(int device_clock_khz, double gate_timeout_seconds)
    : impl_{std::make_unique<Impl>(device_clock_khz, gate_timeout_seconds)} {}
TimingCore::~TimingCore() = default;

std::pair<gamax::column_ptr, gamax::column_ptr> TimingCore::prime_whole(
    PreparedOperation& a, PreparedOperation& b)
{
    return impl_->prime_whole(a, b);
}

Timing TimingCore::run(
    SampleSpec sample, PreparedOperation& a, PreparedOperation& b,
    bool a_first, std::uint64_t a_chunk_size, std::uint64_t b_chunk_size)
{
    return impl_->run(sample, a, b, a_first, a_chunk_size, b_chunk_size);
}

class L2Purger::Impl {
public:
    explicit Impl(int device)
    {
        int l2_size = 0;
        cuda_check(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device),
                   "query L2 cache size");
        bytes_ = static_cast<std::size_t>(std::max(l2_size, 1));
        buffer_.emplace(bytes_, stream_.value, gamax::buffer_init::uninitialized);
    }

    void purge()
    {
        cuda_check(cudaMemsetAsync(buffer_->data(), 0, bytes_, stream_.value), "L2 purge memset");
        cuda_check(cudaStreamSynchronize(stream_.value), "L2 purge synchronization");
    }
    void destroy()
    {
        buffer_.reset();
        cuda_check(cudaStreamSynchronize(stream_.value), "L2 buffer retirement");
    }
    void recreate()
    {
        buffer_.emplace(bytes_, stream_.value, gamax::buffer_init::uninitialized);
    }

    Stream stream_;
    std::size_t bytes_{};
    std::optional<gamax::device_buffer> buffer_;
};

L2Purger::L2Purger(int device) : impl_{std::make_unique<Impl>(device)} {}
L2Purger::~L2Purger() = default;
void L2Purger::purge() { impl_->purge(); }
void L2Purger::destroy() { impl_->destroy(); }
void L2Purger::recreate() { impl_->recreate(); }
std::size_t L2Purger::bytes() const { return impl_->bytes_; }

int gate_self_test(int device, double timeout_seconds)
{
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    int clock_rate_khz = 0;
    cuda_check(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device),
               "query device clock rate");
    Stream a, b;
    SimultaneousGate gate(clock_rate_khz, timeout_seconds);
    Event start_a, stop_a, start_b, stop_b;
    unsigned long long const cycles_a = static_cast<unsigned long long>(clock_rate_khz) * 20;
    unsigned long long const cycles_b = static_cast<unsigned long long>(clock_rate_khz) * 30;
    spin_kernel<<<1, 1, 0, a.value>>>(static_cast<unsigned long long>(clock_rate_khz));
    cuda_check(cudaStreamSynchronize(a.value), "synthetic spin warmup");

    gate.begin();
    gate.block(a.value);
    gate.block(b.value);
    cuda_check(cudaEventRecord(start_a.value, a.value), "self-test A start");
    spin_kernel<<<1, 1, 0, a.value>>>(cycles_a);
    cuda_check(cudaEventRecord(stop_a.value, a.value), "self-test A stop");
    cuda_check(cudaEventRecord(start_b.value, b.value), "self-test B start");
    spin_kernel<<<1, 1, 0, b.value>>>(cycles_b);
    cuda_check(cudaEventRecord(stop_b.value, b.value), "self-test B stop");
    gate.release();
    cuda_check(cudaEventSynchronize(stop_a.value), "self-test A synchronization");
    cuda_check(cudaEventSynchronize(stop_b.value), "self-test B synchronization");
    gate.check();

    double const a_ms = elapsed(start_a.value, stop_a.value);
    double const b_ms = elapsed(start_b.value, stop_b.value);
    float skew = 0.0f;
    cuda_check(cudaEventElapsedTime(&skew, start_a.value, start_b.value), "self-test skew");
    double const skew_ms = std::fabs(static_cast<double>(skew));
    double const makespan = std::max(a_ms, b_ms);
    std::cout << "gate self-test: A=" << a_ms << " ms B=" << b_ms
              << " ms makespan=" << makespan << " ms start_skew=" << skew_ms << " ms\n";
    if (a_ms < 15.0 || a_ms > 35.0 || b_ms < 23.0 || b_ms > 45.0 ||
        makespan > 48.0 || skew_ms > 3.0) {
        std::cerr << "gate self-test failed expected-duration/skew bounds\n";
        return 2;
    }

    SimultaneousGate rejection_gate(clock_rate_khz, 0.05);
    rejection_gate.begin();
    rejection_gate.block(a.value);
    cuda_check(cudaStreamSynchronize(a.value), "hidden-sync rejection synchronization");
    rejection_gate.release();
    bool rejected = false;
    try {
        rejection_gate.check("running deliberate synchronizing adapter");
    } catch (std::runtime_error const&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "gate self-test failed to reject deliberate hidden synchronization\n";
        return 3;
    }
    std::cout << "gate self-test: deliberate hidden synchronization rejected\n";
    return 0;
}

} // namespace gamax::benchmark::concurrency
