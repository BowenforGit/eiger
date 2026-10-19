// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/context.hpp"

#include <cstdlib>
#include <cerrno>
#include <climits>
#include <stdexcept>
#include <string>

#include "memory/mem_manager.hpp"
#include "alps/collectives/fabric_scratch.hpp"
#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/error.hpp"

namespace alps {

void configure_process() {
    // Exclude HPC-X UCC collectives (intermittent MPI_Init segfault). Must be
    // set before MPI_Init reads the MCA environment.
    setenv("OMPI_MCA_coll", "^ucc", /*overwrite=*/0);

    // The legacy backend selector is intentionally unsupported. ALPS uses the
    // same process RMM context as the rest of Eiger; choose arena/direct mode
    // with GAMAX_RMM_RESOURCE before the first allocation.
    if (std::getenv("ALPS_USE_CUDA_MM") != nullptr) {
        throw std::runtime_error(
            "ALPS_USE_CUDA_MM was removed; use GAMAX_RMM_RESOURCE=arena|cuda");
    }
}

Context make_context(MPI_Comm comm) {
    // NVLS (NVLink SHARP multicast) setup aborts inside ncclCommInitRank for
    // >=3 ranks in this container (no fabric-manager/MNNVL support). Disable it
    // by default; ring/tree over NVLink still apply. Overridable: only set if
    // the user has not already chosen a value.
    setenv("NCCL_NVLS_ENABLE", "0", /*overwrite=*/0);

    Context ctx;
    ctx.mpi_comm = comm;
    ALPS_MPICHECK(MPI_Comm_rank(comm, &ctx.rank));
    ALPS_MPICHECK(MPI_Comm_size(comm, &ctx.size));

    int device_count = 0;
    ALPS_CUDACHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
        throw std::runtime_error("alps::make_context: no CUDA device visible");
    }
    int local_rank = -1;
    for (const char* name : {
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID"}) {
        if (const char* value = std::getenv(name)) {
            errno = 0;
            char* end = nullptr;
            long const parsed = std::strtol(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
                throw std::runtime_error(
                    std::string("alps::make_context: invalid ") + name + "='" + value + "'");
            }
            local_rank = static_cast<int>(parsed);
            break;
        }
    }
    if (local_rank < 0) {
        MPI_Comm local_comm = MPI_COMM_NULL;
        ALPS_MPICHECK(MPI_Comm_split_type(
            comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
        ALPS_MPICHECK(MPI_Comm_rank(local_comm, &local_rank));
        ALPS_MPICHECK(MPI_Comm_free(&local_comm));
    }
    if (local_rank >= device_count) {
        throw std::runtime_error(
            "alps::make_context: local rank " + std::to_string(local_rank) +
            " requires a distinct GPU, but only " + std::to_string(device_count) +
            " CUDA device(s) are visible");
    }
    ctx.local_device_id = local_rank;
    ALPS_CUDACHECK(cudaSetDevice(ctx.local_device_id));
    gamax::initialize_device_memory_resource(ctx.local_device_id);
    ALPS_CUDACHECK(cudaStreamCreate(&ctx.stream));

    // Standard MPI+NCCL bootstrap: rank 0 mints the unique id, broadcasts it,
    // then every rank joins the communicator.
    ncclUniqueId id;
    if (ctx.rank == 0) {
        ALPS_NCCLCHECK(ncclGetUniqueId(&id));
    }
    ALPS_MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm));
    ALPS_NCCLCHECK(ncclCommInitRank(&ctx.nccl_comm, ctx.size, id, ctx.rank));

    // Empty (unallocated) CE-collective scratch; the fabric region is created
    // lazily on the first alltoall_column_ce / allgather_column_ce call.
    ctx.fabric_scratch = std::make_shared<collectives::FabricCollectiveScratch>();

    return ctx;
}

void destroy_context(Context& ctx) {
    // Worker streams may retain stream-ordered allocations and must disappear
    // before the communicator and primary stream they coordinate with.
    ctx.column_join_runtime.reset();

    if (ctx.ce_pull_context) {
        ctx.ce_pull_context->free(ctx);
        ctx.ce_pull_context.reset();
    }

    // Free the CE-collective scratch first: its teardown is collective (MPI) and
    // must run while MPI + this rank's device context are still alive.
    if (ctx.fabric_scratch) {
        ctx.fabric_scratch->free(ctx);
        ctx.fabric_scratch.reset();
    }
    if (ctx.nccl_comm != nullptr) {
        ncclCommDestroy(ctx.nccl_comm);
        ctx.nccl_comm = nullptr;
    }
    if (ctx.stream != nullptr) {
        cudaStreamDestroy(ctx.stream);
        ctx.stream = nullptr;
    }
}

} // namespace alps
