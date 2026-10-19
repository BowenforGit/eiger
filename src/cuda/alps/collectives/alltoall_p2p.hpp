// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Direct-P2P all-to-all over the NVLink / MNNVL fabric — an alternative to the
// NCCL path (alltoall_column). Each rank moves its chunks straight across the
// fabric, like nvbandwidth's device_to_device_memcpy_{sm,ce}. Two directions:
//
//   PUSH  — local read  + remote store: write my chunk into peer p's recv buffer.
//   PULL  — remote read + local store:  read peer p's send chunk into my recv
//           (SM only; CE is push-only).
//
// Peer pointers come from the CUDA VMM API + fabric handles: each rank allocates
// its send AND recv buffers with cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC), exports
// the handles, MPI_Allgather exchanges them, and every rank imports + maps every
// peer's send and recv buffers into its own address space (so PUSH can write
// peers' recv and PULL can read peers' send). Works uniformly intra- and
// inter-node (requires the IMEX service). Buffers are allocated ONCE at the
// largest size and reused (each message uses the leading g*M elements).
//
// Layout matches the NCCL CommPattern's uniform pattern: data from sender s
// lands in the receiver's recv buffer at offset s*M, and a rank's send buffer
// holds its chunk for peer p at offset p*M.

#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "types/data_type.hpp"
#include "alps/containers/fabric_peer_buffer.hpp"
#include "alps/context.hpp"

namespace alps::collectives {

/// Copy engine for the transfer: SM = uint4 load/store kernel; CE = cuMemcpyAsync.
enum class P2pCopy { SM, CE };

/// Transfer direction: PUSH (remote store) or PULL (remote load, SM only).
enum class P2pDir { Push, Pull };

/// Fabric-backed send + recv for a positional P2P all-to-all (int32 elements).
/// Both are containers::FabricPeerBuffer (VMM/FABRIC, mapped on every rank via
/// own()/peer(p)/d_peers()), so either direction works.
struct P2pBuffers {
    int               world{0};
    int               rank{0};
    gamax::size_type  send_elems{0};    // send capacity (all2all: g*M_max; allgather: M_max)
    gamax::size_type  recv_elems{0};    // recv capacity (= g*M_max)

    containers::FabricPeerBuffer send; // peers read this (PULL source)
    containers::FabricPeerBuffer recv; // peers write this (PUSH dest)

    // Large fabric copies do not overlap on the GB200 CE. Keep every peer copy
    // on one stream so CUDA cannot reorder the rank-rotated, receiver-balanced
    // peer sequence into a harmful fabric traffic pattern.
    cudaStream_t      ce_stream{};
    cudaEvent_t       ce_done_event{};
    cudaEvent_t       fork_event{};
};

/// Allocate both fabric buffers, exchange handles, map all peers. `send_elems` /
/// `recv_elems` are the per-GPU send/recv capacities (all2all: both = world*M_max;
/// allgather: send = M_max, recv = world*M_max). Collective.
P2pBuffers p2p_alloc(gamax::size_type send_elems, gamax::size_type recv_elems, const Context& ctx);

/// Tear down maps, VA reservations, handles, streams. Collective-safe to call once.
void p2p_free(P2pBuffers& b, const Context& ctx);

/// One positional all-to-all of `per_peer_count` int32 elements per peer, in
/// rotated order (self included). PUSH: send[p*M] -> peer p's recv[rank*M].
/// PULL (SM only): peer p's send[rank*M] -> recv[p*M]. Comm-only (no alloc).
void alltoall_p2p(const P2pBuffers& b, gamax::size_type per_peer_count,
                  P2pCopy copy, P2pDir dir, const Context& ctx, cudaStream_t stream);

} // namespace alps::collectives
