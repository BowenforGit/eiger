// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Direct-P2P all-gather over the NVLink / MNNVL fabric — alternative to the NCCL
// (ncclBroadcast) path. Uniform-count, microbench-grade (mirrors alltoall_p2p).
// Reuses P2pBuffers / P2pCopy / P2pDir / p2p_alloc / p2p_free from alltoall_p2p,
// but with an ASYMMETRIC buffer pair: send holds one chunk (M), recv holds g*M.
//
//   PUSH (sm/ce) — broadcast: write local send[0..M] into every peer p's
//                  recv[rank*M] (the rank's slot in each peer's gather).
//   PULL (sm)    — gather:    read every peer p's send[0..M] into own recv[p*M].
//   self chunk   — local copy, counted.
//
// The only structural difference from alltoall_p2p is the offset: each rank
// contributes ONE chunk that every peer receives, so the send/peer source
// offset is 0 (vs all2all's per-peer rank*M / p*M).

#include "memory/table.hpp"
#include "alps/collectives/alltoall_p2p.hpp" // P2pBuffers, P2pCopy, P2pDir, p2p_alloc/free
#include "alps/context.hpp"

namespace alps::collectives {

/// One positional all-gather of `per_peer_count` (M) int32 elements per rank,
/// rotated peer order (self included). PUSH: send[0..M] -> peer p's recv[rank*M].
/// PULL (SM only): peer p's send[0..M] -> recv[p*M]. CE supports push only.
/// Allocate `b` with p2p_alloc(send_elems = M_max, recv_elems = g*M_max).
void allgather_p2p(const P2pBuffers& b, gamax::size_type per_peer_count,
                   P2pCopy copy, P2pDir dir, const Context& ctx, cudaStream_t stream);

} // namespace alps::collectives
