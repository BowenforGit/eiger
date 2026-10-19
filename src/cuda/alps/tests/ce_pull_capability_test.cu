// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// Capability and ordering test for the CE-pull transport prerequisites.
//
// This is intentionally lower-level than a shuffle correctness test. It proves
// on the actual MPI/MNNVL topology that:
//   1. cudaMemcpyBatchAsync can read imported fabric VMM and write ordinary RMM;
//   2. 64-bit CUDA stream-memory writes/waits work on fabric-mapped doorbells;
//   3. the one-warp polling fallback observes the same remote writes; and
//   4. two slots can be reused with monotonically increasing epochs.

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/containers/fabric_peer_buffer.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"
#include "memory/device_buffer.hpp"

namespace {

constexpr int         kSlots          = 2;
constexpr int         kProtocolEpochs = 32;
constexpr std::size_t kWordsPerPeer   = 256;
constexpr std::size_t kBytesPerPeer   = kWordsPerPeer * sizeof(std::uint64_t);

std::uint64_t expected_word(int sender, int receiver, std::size_t index) {
    return (static_cast<std::uint64_t>(sender) << 48) |
           (static_cast<std::uint64_t>(receiver) << 32) |
           static_cast<std::uint64_t>(index);
}

int aggregate_failure(int local_fail, const alps::Context& ctx) {
    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    return global_fail;
}

__global__ void poll_geq_kernel(const unsigned long long* words, int count,
                                unsigned long long epoch) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    for (int i = lane; i < count; i += 32) {
        unsigned long long value = 0;
        do {
            value = __ldcg(words + i);
#if __CUDA_ARCH__ >= 700
            if (static_cast<long long>(value - epoch) < 0) __nanosleep(64);
#endif
        } while (static_cast<long long>(value - epoch) < 0);
    }
}

CUdeviceptr word_address(CUdeviceptr base, std::size_t index) {
    return base + index * sizeof(unsigned long long);
}

std::size_t ready_index(int world, int slot, int sender) {
    return static_cast<std::size_t>(slot) * 2 * world + sender;
}

std::size_t consumed_index(int world, int slot, int receiver) {
    return static_cast<std::size_t>(slot) * 2 * world + world + receiver;
}

int verify_batch_pull(alps::containers::FabricPeerBuffer& source,
                      const alps::Context& ctx) {
    const std::size_t source_bytes = static_cast<std::size_t>(ctx.size) * kBytesPerPeer;
    source.ensure(source_bytes, ctx);

    std::vector<std::uint64_t> local_source(source_bytes / sizeof(std::uint64_t));
    for (int receiver = 0; receiver < ctx.size; ++receiver) {
        for (std::size_t i = 0; i < kWordsPerPeer; ++i) {
            local_source[static_cast<std::size_t>(receiver) * kWordsPerPeer + i] =
                expected_word(ctx.rank, receiver, i);
        }
    }
    ALPS_CUDACHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(source.own())),
        local_source.data(), source_bytes, cudaMemcpyHostToDevice, ctx.stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));

    gamax::device_buffer output(source_bytes, ctx.stream, gamax::buffer_init::zero);
    std::vector<void*> destinations(ctx.size);
    std::vector<const void*> sources(ctx.size);
    std::vector<std::size_t> sizes(ctx.size, kBytesPerPeer);
    for (int sender = 0; sender < ctx.size; ++sender) {
        destinations[sender] = static_cast<char*>(output.data()) +
                               static_cast<std::size_t>(sender) * kBytesPerPeer;
        sources[sender] = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(
            source.peer(sender) + static_cast<std::size_t>(ctx.rank) * kBytesPerPeer));
    }

    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attributes.flags          = cudaMemcpyFlagPreferOverlapWithCompute;
    std::size_t attributes_index = 0;
    ALPS_CUDACHECK(cudaMemcpyBatchAsync(
        destinations.data(), sources.data(), sizes.data(), destinations.size(),
        &attributes, &attributes_index, 1, ctx.stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    std::vector<std::uint64_t> host_output(local_source.size());
    ALPS_CUDACHECK(cudaMemcpy(host_output.data(), output.data(), source_bytes,
                              cudaMemcpyDeviceToHost));
    int failed = 0;
    for (int sender = 0; sender < ctx.size && !failed; ++sender) {
        for (std::size_t i = 0; i < kWordsPerPeer; ++i) {
            const auto got = host_output[
                static_cast<std::size_t>(sender) * kWordsPerPeer + i];
            const auto want = expected_word(sender, ctx.rank, i);
            if (got != want) {
                std::fprintf(stderr,
                    "[rank %d] batch pull mismatch sender=%d index=%zu "
                    "got=0x%llx want=0x%llx\n",
                    ctx.rank, sender, i,
                    static_cast<unsigned long long>(got),
                    static_cast<unsigned long long>(want));
                failed = 1;
                break;
            }
        }
    }
    return failed;
}

enum class WaitMode { StreamMemOps, PollKernel };

int verify_doorbells(alps::containers::FabricPeerBuffer& control,
                     const alps::Context& ctx, WaitMode mode) {
    const std::size_t words = static_cast<std::size_t>(kSlots) * 2 * ctx.size;
    control.ensure(words * sizeof(unsigned long long), ctx);
    ALPS_CUDACHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(control.own())), 0,
        words * sizeof(unsigned long long), ctx.stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));

    cudaStream_t signal_stream{};
    cudaStream_t transfer_stream{};
    cudaStream_t reclaim_stream{};
    ALPS_CUDACHECK(cudaStreamCreateWithFlags(&signal_stream, cudaStreamNonBlocking));
    ALPS_CUDACHECK(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
    ALPS_CUDACHECK(cudaStreamCreateWithFlags(&reclaim_stream, cudaStreamNonBlocking));

    cudaEvent_t slot_free[kSlots]{};
    cudaEvent_t ready_sent[kSlots]{};
    cudaEvent_t consumed_sent[kSlots]{};
    for (int slot = 0; slot < kSlots; ++slot) {
        ALPS_CUDACHECK(cudaEventCreateWithFlags(
            &slot_free[slot], cudaEventDisableTiming));
        ALPS_CUDACHECK(cudaEventCreateWithFlags(
            &ready_sent[slot], cudaEventDisableTiming));
        ALPS_CUDACHECK(cudaEventCreateWithFlags(
            &consumed_sent[slot], cudaEventDisableTiming));
        ALPS_CUDACHECK(cudaEventRecord(slot_free[slot], ctx.stream));
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    const auto* local_words = reinterpret_cast<const unsigned long long*>(
        static_cast<std::uintptr_t>(control.own()));

    for (int op = 0; op < kProtocolEpochs; ++op) {
        const int slot = op % kSlots;
        const auto epoch = static_cast<unsigned long long>(op + 1);

        // CUDA-visible local progress edge: a rank publishes all of its READY
        // signals before its transfer stream starts waiting for peers.
        ALPS_CUDACHECK(cudaStreamWaitEvent(signal_stream, slot_free[slot], 0));
        for (int receiver = 0; receiver < ctx.size; ++receiver) {
            ALPS_CUCHECK(cuStreamWriteValue64(
                reinterpret_cast<CUstream>(signal_stream),
                word_address(control.peer(receiver),
                             ready_index(ctx.size, slot, ctx.rank)),
                epoch, CU_STREAM_WRITE_VALUE_DEFAULT));
        }
        ALPS_CUDACHECK(cudaEventRecord(ready_sent[slot], signal_stream));

        ALPS_CUDACHECK(cudaStreamWaitEvent(
            transfer_stream, ready_sent[slot], 0));
        const auto ready_base = local_words + ready_index(ctx.size, slot, 0);
        if (mode == WaitMode::StreamMemOps) {
            for (int sender = 0; sender < ctx.size; ++sender) {
                ALPS_CUCHECK(cuStreamWaitValue64(
                    reinterpret_cast<CUstream>(transfer_stream),
                    word_address(control.own(),
                                 ready_index(ctx.size, slot, sender)),
                    epoch, CU_STREAM_WAIT_VALUE_GEQ));
            }
        } else {
            poll_geq_kernel<<<1, 32, 0, transfer_stream>>>(
                ready_base, ctx.size, epoch);
            ALPS_CUDACHECK(cudaGetLastError());
        }

        // This rank is the receiver. Acknowledge every sender after the wait.
        for (int sender = 0; sender < ctx.size; ++sender) {
            ALPS_CUCHECK(cuStreamWriteValue64(
                reinterpret_cast<CUstream>(transfer_stream),
                word_address(control.peer(sender),
                             consumed_index(ctx.size, slot, ctx.rank)),
                epoch, CU_STREAM_WRITE_VALUE_DEFAULT));
        }
        ALPS_CUDACHECK(cudaEventRecord(consumed_sent[slot], transfer_stream));

        // Another CUDA-visible progress edge ensures local ACK publication is
        // schedulable before this rank waits for ACKs from peers.
        ALPS_CUDACHECK(cudaStreamWaitEvent(
            reclaim_stream, consumed_sent[slot], 0));
        const auto consumed_base =
            local_words + consumed_index(ctx.size, slot, 0);
        if (mode == WaitMode::StreamMemOps) {
            for (int receiver = 0; receiver < ctx.size; ++receiver) {
                ALPS_CUCHECK(cuStreamWaitValue64(
                    reinterpret_cast<CUstream>(reclaim_stream),
                    word_address(control.own(),
                                 consumed_index(ctx.size, slot, receiver)),
                    epoch, CU_STREAM_WAIT_VALUE_GEQ));
            }
        } else {
            poll_geq_kernel<<<1, 32, 0, reclaim_stream>>>(
                consumed_base, ctx.size, epoch);
            ALPS_CUDACHECK(cudaGetLastError());
        }
        ALPS_CUDACHECK(cudaEventRecord(slot_free[slot], reclaim_stream));
    }

    ALPS_CUDACHECK(cudaStreamSynchronize(reclaim_stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(transfer_stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(signal_stream));

    std::vector<unsigned long long> observed(words);
    ALPS_CUDACHECK(cudaMemcpy(
        observed.data(),
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(control.own())),
        words * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    int failed = 0;
    for (int slot = 0; slot < kSlots; ++slot) {
        int last_op = kProtocolEpochs - 1;
        if (last_op % kSlots != slot) --last_op;
        const auto expected = static_cast<unsigned long long>(last_op + 1);
        for (int peer = 0; peer < ctx.size; ++peer) {
            const auto ready = observed[ready_index(ctx.size, slot, peer)];
            const auto consumed = observed[consumed_index(ctx.size, slot, peer)];
            if (ready != expected || consumed != expected) {
                std::fprintf(stderr,
                    "[rank %d] %s doorbell mismatch slot=%d peer=%d "
                    "ready=%llu consumed=%llu want=%llu\n",
                    ctx.rank,
                    mode == WaitMode::StreamMemOps ? "stream" : "poll",
                    slot, peer, ready, consumed, expected);
                failed = 1;
            }
        }
    }

    for (int slot = 0; slot < kSlots; ++slot) {
        cudaEventDestroy(consumed_sent[slot]);
        cudaEventDestroy(ready_sent[slot]);
        cudaEventDestroy(slot_free[slot]);
    }
    cudaStreamDestroy(reclaim_stream);
    cudaStreamDestroy(transfer_stream);
    cudaStreamDestroy(signal_stream);
    return failed;
}

} // namespace

int main(int argc, char** argv) {
    bool inject_asymmetric_failure_after_batch = false;
    bool inject_asymmetric_vmm_allocation_failure = false;
    if (argc == 2 && std::strcmp(
            argv[1], "--inject-asymmetric-failure-after-batch") == 0) {
        inject_asymmetric_failure_after_batch = true;
    } else if (argc == 2 && std::strcmp(
            argv[1], "--inject-asymmetric-vmm-allocation-failure") == 0) {
        inject_asymmetric_vmm_allocation_failure = true;
    } else if (argc != 1) {
        std::fprintf(
            stderr,
            "usage: %s [--inject-asymmetric-failure-after-batch|"
            "--inject-asymmetric-vmm-allocation-failure]\n",
            argv[0]);
        return 2;
    }

    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    CUdevice device{};
    ALPS_CUCHECK(cuDeviceGet(&device, ctx.local_device_id));
    int fabric_supported = 0;
    int stream64_supported = 0;
    ALPS_CUCHECK(cuDeviceGetAttribute(
        &fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        device));
    ALPS_CUCHECK(cuDeviceGetAttribute(
        &stream64_supported,
        CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device));

    int local_fail = (!fabric_supported || !stream64_supported) ? 1 : 0;
    if (local_fail) {
        std::fprintf(stderr,
            "[rank %d] capability missing: fabric=%d stream_memops64=%d\n",
            ctx.rank, fabric_supported, stream64_supported);
    }

    // Branch only on rank-wide phase status. A rank-local mismatch or missing
    // capability must make every rank skip the following collective phase.
    const int capability_fail = aggregate_failure(local_fail, ctx);
    int batch_verification_fail = capability_fail;
    int global_fail = capability_fail;
    alps::containers::FabricPeerBuffer source;
    alps::containers::FabricPeerBuffer control;
    if (!global_fail && inject_asymmetric_vmm_allocation_failure) {
        setenv(
            "ALPS_TEST_FABRIC_PEER_BUFFER_FAIL_STAGE",
            "after_local_export", 1);
        int did_not_throw = 0;
        try {
            source.ensure(kBytesPerPeer, ctx);
            did_not_throw = 1;
        } catch (...) {
            // Every rank must leave ensure through an exception instead of
            // entering a mismatched handle exchange or later collective.
        }
        unsetenv("ALPS_TEST_FABRIC_PEER_BUFFER_FAIL_STAGE");
        global_fail = aggregate_failure(did_not_throw, ctx);
    }
    if (!global_fail && !inject_asymmetric_vmm_allocation_failure) {
        batch_verification_fail = aggregate_failure(
            verify_batch_pull(source, ctx), ctx);
        global_fail = batch_verification_fail;
    }
    if (!global_fail && inject_asymmetric_failure_after_batch) {
        global_fail = aggregate_failure(ctx.rank == 0 ? 1 : 0, ctx);
    }
    if (!global_fail && !inject_asymmetric_vmm_allocation_failure) {
        global_fail = aggregate_failure(
            verify_doorbells(control, ctx, WaitMode::StreamMemOps), ctx);
    }
    if (!global_fail && !inject_asymmetric_vmm_allocation_failure) {
        global_fail = aggregate_failure(
            verify_doorbells(control, ctx, WaitMode::PollKernel), ctx);
    }

    const int result_fail = inject_asymmetric_failure_after_batch
        ? (capability_fail || batch_verification_fail || !global_fail)
        : global_fail;
    if (ctx.rank == 0) {
        std::printf(
            "[ce_pull_capability_test] np=%d batch_pull+stream_memops+poll "
            "epochs=%d%s -> %s\n",
            ctx.size, kProtocolEpochs,
            inject_asymmetric_failure_after_batch
                ? "+coordinated-asymmetric-failure"
                : (inject_asymmetric_vmm_allocation_failure
                    ? "+coordinated-vmm-allocation-failure" : ""),
            result_fail ? "FAIL" : "PASS");
    }

    control.free(ctx);
    source.free(ctx);
    alps::destroy_context(ctx);
    MPI_Finalize();
    return result_fail;
}
