// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

namespace alps::collectives {

enum class CeDoorbellMode {
    StreamMemOps,
    // Replaces stream-memory waits with a one-warp polling kernel. READY and
    // completion publication still use cuStreamWriteValue64, so this mode also
    // requires 64-bit stream-memory-operation support.
    PollKernel,
};

// cudaMemcpyBatchAsync does not specify descriptor execution order. These
// modes retain the original one-batch path and descriptor-order diagnostic
// alongside the selected production policy. RotatedDescriptors changes only
// descriptor order; RoundRobinWaves submits one cyclic sender matching per
// batch and is the CE-pull default.
enum class CePullSchedule {
    Unordered,
    RotatedDescriptors,
    RoundRobinWaves,
};

struct CePullOptions {
    int slots {2};
    CeDoorbellMode doorbell_mode {CeDoorbellMode::StreamMemOps};
    CePullSchedule schedule {CePullSchedule::RoundRobinWaves};
    bool high_priority_transfer_stream {true};
    bool prefer_copy_compute_overlap {true};
    // Full-staging batches assign one distinct arena lease and doorbell slot
    // to every live logical column. Legacy/default collectives use the bounded
    // operation%slots ring.
    bool full_staging {false};
};

} // namespace alps::collectives
