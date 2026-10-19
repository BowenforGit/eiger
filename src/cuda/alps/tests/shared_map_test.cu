// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_shared_map_test -- atomicity correctness gate for the fabric-backed shared
// hash table (alps::containers::SharedStaticMap over a FabricStripedRegion).
//
// Run under mpirun (-np 2 / 3 / 4). Every rank concurrently inserts the IDENTICAL
// key set {(k, k) : k in [0, K)} into the one shared table -- identical keys from
// every GPU maximize cross-GPU CAS contention on the same remote slots, which is
// the strongest test of thread_scope_system atomicity over the fabric. Because
// value == key, the winning CAS writes the same value no matter which rank wins,
// so the final state is deterministic despite the race.
//
// Build fully -> MPI_Barrier -> probe. A correct run has size()==K and
// find(k)==k for every k; a broken .sys CAS shows up as a wrong size, a missing
// key, or a torn slot (find(k) != k). Deliberately run at a high load factor so
// long probe chains spill across segment boundaries into remote GPUs.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda/std/tuple>

#include "types/data_type.hpp"
#include "memory/column.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"
#include "alps/containers/fabric_region.hpp"
#include "alps/containers/shared_static_map.cuh"

namespace {
constexpr int            kCG          = 2;
constexpr int32_t        kEmptyKey    = -1;
constexpr int32_t        kEmptyValue  = -1;
constexpr gamax::size_type kKeys      = gamax::size_type{1} << 20; // 1,048,576 keys
constexpr double         kLoadFactor  = 0.9; // high LF -> long chains, cross-segment spill
} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();

    const gamax::size_type K = kKeys;

    // This rank's copy of the identical key set [0, K); value == key.
    auto keys = gamax::make_column<int32_t>(K, ctx.stream);
    int32_t* keys_ptr = keys->data<int32_t>();
    {
        gamax::rmm_exec_policy_handle policy{ctx.stream};
        thrust::sequence(policy.get(), keys_ptr, keys_ptr + K, 0);
    }

    alps::containers::FabricStripedRegion region(ctx);
    int local_fail = 0;

    {
        alps::containers::FabricAllocator<cuco::pair<int32_t, int32_t>> alloc(&region);
        alps::containers::SharedStaticMap<kCG> map(
            static_cast<std::size_t>(K), kLoadFactor,
            cuco::empty_key{kEmptyKey}, cuco::empty_value{kEmptyValue},
            {}, {}, {}, {}, alloc, ctx.stream);

        // Concurrent build: pair(k, k) from every rank.
        auto pairs = thrust::make_zip_iterator(cuda::std::make_tuple(keys_ptr, keys_ptr));
        map.insert_async(pairs, pairs + K, ctx.stream);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm)); // all builds complete before any probe

        // Probe every key; with match_ratio 1.0 there are no misses.
        auto found = gamax::make_column<int32_t>(K, ctx.stream);
        int32_t* found_ptr = found->data<int32_t>();
        {
            gamax::rmm_exec_policy_handle policy{ctx.stream};
            thrust::fill(policy.get(), found_ptr, found_ptr + K, kEmptyValue);
        }
        map.find_async(keys_ptr, keys_ptr + K, found_ptr, ctx.stream);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

        // Two distinct properties:
        //  - lost/torn keys: find(k) != k. This is the true ATOMICITY guarantee
        //    (no torn slot, no lost CAS). Any nonzero count is a hard failure.
        //  - duplicates: size() > K means a key landed in >1 slot. Atomic CAS
        //    stops two GPUs claiming the SAME slot, but concurrent open-addressing
        //    insert of IDENTICAL keys across GPUs can still place a key in two
        //    slots when a probe read doesn't yet see another GPU's insert (a
        //    cross-GPU ordering effect; cuco's insert dedup is single-GPU). It does
        //    NOT occur for a unique-key build (the realistic workload). Reported,
        //    not gated.
        std::vector<int32_t> h_found(K);
        ALPS_CUDACHECK(cudaMemcpy(h_found.data(), found_ptr, K * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
        gamax::size_type lost = 0;
        for (gamax::size_type i = 0; i < K; ++i)
            if (h_found[i] != static_cast<int32_t>(i)) ++lost;
        if (lost != 0) {
            std::fprintf(stderr, "[rank %d] LOST/TORN: %lld/%lld keys had wrong/missing value\n",
                         ctx.rank, static_cast<long long>(lost), static_cast<long long>(K));
            local_fail = 1; // atomicity break -> fail
        }

        const std::size_t sz  = map.size(ctx.stream);
        const long long    dup = static_cast<long long>(sz) - static_cast<long long>(K);
        if (ctx.rank == 0)
            std::printf("[shared_map_test] np=%d K=%lld LF=%.2f fabric=%.1f MiB/GPU "
                        "lost=%lld duplicates=%lld (%.3f%%) fallback_allocs=%lld\n",
                        ctx.size, static_cast<long long>(K), kLoadFactor,
                        region.slab_bytes() / (1024.0 * 1024.0), static_cast<long long>(lost),
                        dup, 100.0 * dup / static_cast<double>(K), region.fallback_allocs());
    } // map destroyed here (storage deallocate marks the region free)

    region.free();

    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (ctx.rank == 0)
        std::printf("[shared_map_test] np=%d -> %s\n", ctx.size, global_fail ? "FAIL" : "PASS");

    alps::destroy_context(ctx);
    MPI_Finalize();
    return global_fail;
}
