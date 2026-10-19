// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// alps::Context bundles the per-rank distributed-execution state: the CUDA
// stream all device work is issued on, the NCCL communicator, the MPI
// communicator, and this rank's identity / bound device.
//
// One MPI rank == one process == one GPU. MPI_Init must already have been
// called (by the executable's main) before make_context().

#include <memory>

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>

#include "alps/collectives/ce_pull_options.hpp"

namespace gamax::exec { class StepProfileSink; }

namespace alps {

namespace collectives {
struct FabricCollectiveScratch;
class CePullExchangeContext;
}
namespace operators { struct ColumnJoinRuntime; }

struct Context {
    cudaStream_t stream{};
    ncclComm_t   nccl_comm{};
    MPI_Comm     mpi_comm{MPI_COMM_WORLD};
    int          rank{0};
    int          size{1};
    int          local_device_id{0};

    /// Persistent scratch for the copy-engine fabric collectives (CE backend):
    /// the reused fabric recv staging region + its sub-streams/events. Created
    /// (empty, unallocated) in make_context; the fabric region is allocated
    /// lazily on the first CE collective and freed in destroy_context. Non-null
    /// after make_context, so const-ref call sites only ever mutate the pointee.
    std::shared_ptr<collectives::FabricCollectiveScratch> fabric_scratch;

    // Lazily created by the opt-in CE-pull shuffle. It owns the persistent
    // sender VMM arena, stream-ordered doorbells, and global slot timeline.
    mutable std::shared_ptr<collectives::CePullExchangeContext> ce_pull_context;

    // Default policy for blocking/global CE-pull collectives. Distributed
    // execution configures this once per run; ColumnOverlap makes a copy and
    // enables full_staging for its operator-local overlap schedule.
    mutable collectives::CePullOptions ce_pull_options;

    // Lazily created only by the column-overlapped equality-join alternative.
    // Mutable because distributed operators receive Context by const reference
    // while this cache does not alter rank/communicator semantics.
    mutable std::shared_ptr<operators::ColumnJoinRuntime> column_join_runtime;

    // Non-owning sink enabled only for the dedicated NCCL message-trace pass.
    // Ordinary timing, breakdown, memory, and profile runs leave it null.
    mutable gamax::exec::StepProfileSink* message_trace_sink {nullptr};

    // Non-copyable: a Context owns an NCCL communicator, a CUDA stream, and the
    // shared fabric scratch, all torn down exactly once by destroy_context(). A
    // shallow copy would alias those and double-free / double-run the COLLECTIVE
    // scratch free() (an extra unmatched MPI_Barrier -> desync). Movable so
    // make_context() can return by value.
    Context()                          = default;
    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&)                 = default;
    Context& operator=(Context&&)      = default;

    /// True when the world size is a power of two. The shuffle no longer
    /// requires this (it range-reduces the hash into [0, size) for any size);
    /// kept as a convenience for callers that still want to assert it.
    bool is_pow2_size() const noexcept { return size > 0 && (size & (size - 1)) == 0; }
    /// Bits needed to address all ranks: ceil(log2(size)). The shuffle radix-
    /// partitions destinations into 2^rank_bits buckets (the high ones empty
    /// when size is not a power of two).
    int  rank_bits()    const noexcept { int b = 0; while ((1 << b) < size) ++b; return b; }
};

/// One-shot alps process setup. MUST be called first in main(), before
/// MPI_Init and before any gamax allocation. Applies the MPI default below:
///
///  - OMPI_MCA_coll=^ucc: the HPC-X UCC collective component intermittently
///    segfaults allocating host shared memory inside MPI_Init on this system.
///    Our MPI collectives are tiny, so excluding UCC (falling back to tuned/
///    basic) is free. Both env defaults use overwrite=0 — set them yourself to
///    override.
void configure_process();

/// Bind this local rank to its same-numbered GPU, prewarm the process RMM
/// resource, create a stream, and build a NCCL communicator. Oversubscription
/// is rejected: ALPS requires one visible GPU per local rank.
Context make_context(MPI_Comm comm = MPI_COMM_WORLD);

/// Destroy the NCCL communicator and CUDA stream. Safe to call once.
void destroy_context(Context& ctx);

} // namespace alps
