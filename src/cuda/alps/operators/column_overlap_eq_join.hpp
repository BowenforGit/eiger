// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "alps/collectives/alltoall.hpp"
#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/context.hpp"
#include "memory/table.hpp"
#include "operators/join/join.hpp"

namespace gamax::exec { class StepProfileSink; }

namespace alps::operators {

enum class ColumnSchedule {
    ProbeFirst,
    BuildFirst,
    Interleave,
};

struct ColumnOverlapTiming {
    float total_ms {0.0f};
    float key_prepare_ms {0.0f};
    float payload_prepare_ms {0.0f};
    float key_exchange_ms {0.0f};
    float payload_exchange_ms {0.0f};
    float match_ms {0.0f};
    float materialize_ms {0.0f};
};

// Lazily owned by alps::Context. The distributed executor is sequential, so
// one runtime has at most one active column-overlap join.
struct ColumnJoinRuntime {
    cudaStream_t communication_stream {nullptr};
    cudaStream_t preparation_stream {nullptr};
    cudaStream_t join_stream {nullptr};
    bool communication_high_priority {false};
    std::vector<cudaEvent_t> dependency_events;

    explicit ColumnJoinRuntime(bool communication_high_priority = true);
    ~ColumnJoinRuntime();
    ColumnJoinRuntime(const ColumnJoinRuntime&) = delete;
    ColumnJoinRuntime& operator=(const ColumnJoinRuntime&) = delete;
    void ensure_dependency_events(std::size_t count);
};

// Column-overlapped distributed HASH equality join. Keys are fixed-width;
// payloads may be fixed-width or Arrow-layout STRING columns. A side is hash
// shuffled only when its partition flag is set; otherwise it must already have
// the required distribution. Inputs and outputs follow phys::eq_join semantics:
// [shared key, left non-keys in input order, right non-keys in input order].
std::unique_ptr<gamax::table> column_overlap_eq_join(
    gamax::table_view left,
    gamax::size_type left_key,
    gamax::table_view right,
    gamax::size_type right_key,
    const gamax::join::join_hint& hint,
    const alps::Context& context,
    cudaStream_t caller_stream,
    bool partition_left = true,
    bool partition_right = true,
    ColumnSchedule schedule = ColumnSchedule::Interleave,
    ColumnOverlapTiming* timing = nullptr,
    gamax::exec::StepProfileSink* profile_sink = nullptr,
    bool communication_high_priority = true,
    bool early_reclaim = false,
    alps::collectives::Backend exchange_backend =
        alps::collectives::Backend::NCCL,
    alps::collectives::CePullSchedule ce_pull_schedule =
        alps::collectives::CePullSchedule::RoundRobinWaves,
    int ce_pull_slots = 2);

} // namespace alps::operators
