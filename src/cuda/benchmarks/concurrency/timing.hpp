// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "operation.hpp"

namespace gamax::benchmark::concurrency {

enum class Mode { a_alone, b_alone, sequential_ab, sequential_ba, concurrent };
enum class ExecutionVariant { whole, chunked };

struct SampleSpec {
    Mode mode{};
    ExecutionVariant variant{ExecutionVariant::whole};
};

inline bool operator==(SampleSpec lhs, SampleSpec rhs)
{
    return lhs.mode == rhs.mode && lhs.variant == rhs.variant;
}

inline bool operator!=(SampleSpec lhs, SampleSpec rhs) { return !(lhs == rhs); }

char const* mode_name(Mode mode);
char const* variant_name(ExecutionVariant variant);
std::string sample_name(SampleSpec sample);
std::vector<SampleSpec> all_samples(bool include_chunked);

class ScopedSampleNvtxRange {
public:
    explicit ScopedSampleNvtxRange(SampleSpec sample);
    ~ScopedSampleNvtxRange();
    ScopedSampleNvtxRange(ScopedSampleNvtxRange const&) = delete;
    ScopedSampleNvtxRange& operator=(ScopedSampleNvtxRange const&) = delete;
};

struct Timing {
    double a_ms{};
    double b_ms{};
    double makespan_ms{};
    double start_skew_ms{};
    std::string order;
    gamax::column_ptr a_result;
    gamax::column_ptr b_result;
};

class TimingCore {
public:
    TimingCore(int device_clock_khz, double gate_timeout_seconds);
    ~TimingCore();
    TimingCore(TimingCore const&) = delete;
    TimingCore& operator=(TimingCore const&) = delete;

    std::pair<gamax::column_ptr, gamax::column_ptr> prime_whole(
        PreparedOperation& a, PreparedOperation& b);
    Timing run(
        SampleSpec sample, PreparedOperation& a, PreparedOperation& b,
        bool a_first, std::uint64_t a_chunk_size, std::uint64_t b_chunk_size);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class L2Purger {
public:
    explicit L2Purger(int device);
    ~L2Purger();
    L2Purger(L2Purger const&) = delete;
    L2Purger& operator=(L2Purger const&) = delete;

    void purge();
    void destroy();
    void recreate();
    std::size_t bytes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

int gate_self_test(int device, double timeout_seconds);

} // namespace gamax::benchmark::concurrency
