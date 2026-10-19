// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Per-Context persistent scratch for the copy-engine fabric collectives
// (alltoall_column_ce / allgather_column_ce). It owns:
//
//   - recv: a FabricPeerBuffer used as the PUSH destination. Every rank's recv is
//     mapped on every peer, so a peer's copy engine writes its chunk straight
//     into this rank's recv across NVLink. Sized (grow-only) to the largest
//     total_recv seen, reused across all columns of a table and across calls.
//   - sub_streams/join_events/fork_event: the fork/join fan-out so the per-peer
//     cuMemcpyAsync pushes run on concurrent copy engines.
//   - barrier_buf: a 4-byte device scratch for the trailing stream-ordered NCCL
//     all-reduce that fences cross-rank completion before the copy-out.
//
// Owned by Context via shared_ptr (created empty in make_context, the fabric
// region allocated lazily on first CE collective, freed in destroy_context
// before MPI_Finalize). Stateless call sites pull it out of ctx, exactly like
// they pull ctx.nccl_comm.

#include <vector>

#include <cuda_runtime_api.h>

#include "alps/containers/fabric_peer_buffer.hpp"
#include "alps/context.hpp"
#include "memory/device_buffer.hpp"

namespace alps::collectives {

// NOT reentrant: the single recv buffer + fork/join events + sub-streams below are
// shared per Context, so only ONE CE collective may be in flight at a time and all
// CE collectives on a given Context must be issued on the same (caller's) stream —
// which every current caller does (ctx.stream, sequentially).
struct FabricCollectiveScratch {
    containers::FabricPeerBuffer recv;        // PUSH destination, all peers mapped
    std::vector<cudaStream_t>    sub_streams; // [world] concurrent CE copies
    std::vector<cudaEvent_t>     join_events; // [world] join back to caller stream
    cudaEvent_t                  fork_event{};
    gamax::device_buffer         barrier_buf{};       // 4B: in-place NCCL barrier
    bool                         streams_inited{false};

    FabricCollectiveScratch() = default;
    /// Best-effort LOCAL reclaim of the streams/events/barrier buffer (and recv's
    /// own local teardown) in case the owning Context is dropped without calling
    /// destroy_context(); a no-op after free() ran. Cannot do the COLLECTIVE fabric
    /// teardown (no MPI in a destructor) — callers must still free() before
    /// MPI_Finalize for an orderly, barrier-synchronized teardown.
    ~FabricCollectiveScratch();

    /// Lazily create the `world` sub-streams + fork/join events + barrier buffer
    /// (no-op after the first call). Not collective.
    void ensure_streams(int world, cudaStream_t stream);

    /// COLLECTIVE teardown of the fabric region + local teardown of the streams/
    /// events/barrier buffer. Idempotent; call once before MPI_Finalize.
    void free(const Context& ctx);
};

} // namespace alps::collectives
