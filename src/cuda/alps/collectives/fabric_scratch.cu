// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/fabric_scratch.hpp"

#include <cuda_runtime.h>

#include "alps/error.hpp"

namespace alps::collectives {

FabricCollectiveScratch::~FabricCollectiveScratch() {
    // Local-only reclaim (no MPI): safe whether or not free() already ran — after
    // free() the vectors are empty and the handles null, so this is a no-op. The
    // recv member's ~FabricPeerBuffer does its own local teardown automatically.
    for (auto s : sub_streams) if (s) cudaStreamDestroy(s);
    for (auto e : join_events) if (e) cudaEventDestroy(e);
    if (fork_event)  cudaEventDestroy(fork_event);
}

void FabricCollectiveScratch::ensure_streams(int world, cudaStream_t stream) {
    if (streams_inited) return;
    sub_streams.resize(world);
    join_events.resize(world);
    for (int i = 0; i < world; ++i) {
        ALPS_CUDACHECK(cudaStreamCreateWithFlags(&sub_streams[i], cudaStreamNonBlocking));
        ALPS_CUDACHECK(cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming));
    }
    ALPS_CUDACHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
    barrier_buf = gamax::device_buffer(
        sizeof(int), stream, gamax::buffer_init::zero);
    streams_inited = true;
}

void FabricCollectiveScratch::free(const Context& ctx) {
    recv.free(ctx); // collective (MPI_Barrier inside)
    for (auto s : sub_streams) if (s) cudaStreamDestroy(s);
    for (auto e : join_events) if (e) cudaEventDestroy(e);
    if (fork_event)  { cudaEventDestroy(fork_event); fork_event = nullptr; }
    barrier_buf = gamax::device_buffer{};
    sub_streams.clear();
    join_events.clear();
    streams_inited = false;
}

} // namespace alps::collectives
