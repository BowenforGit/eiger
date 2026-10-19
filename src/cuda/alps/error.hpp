// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Error-checking macros for the ALPS multi-GPU component. Each macro aborts
// the process (with the MPI rank for context) on failure. Kept deliberately
// blunt: a failed collective or CUDA call leaves the job unrecoverable, so
// there is nothing useful to do but report and exit.

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>

namespace alps {
namespace detail {
inline int current_mpi_rank() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) return -1;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}
} // namespace detail
} // namespace alps

#ifndef ALPS_CUDACHECK
#define ALPS_CUDACHECK(cmd)                                                                    \
    do {                                                                                       \
        cudaError_t _e = (cmd);                                                                \
        if (_e != cudaSuccess) {                                                               \
            std::fprintf(stderr, "[rank %d] CUDA error %s:%d '%s'\n",                           \
                         alps::detail::current_mpi_rank(), __FILE__, __LINE__,                  \
                         cudaGetErrorString(_e));                                              \
            std::abort();                                                                       \
        }                                                                                      \
    } while (0)
#endif

#ifndef ALPS_CUCHECK
#define ALPS_CUCHECK(cmd)                                                                      \
    do {                                                                                       \
        CUresult _c = (cmd);                                                                   \
        if (_c != CUDA_SUCCESS) {                                                              \
            const char* _s = nullptr;                                                          \
            cuGetErrorString(_c, &_s);                                                         \
            std::fprintf(stderr, "[rank %d] CU driver error %s:%d '%s'\n",                      \
                         alps::detail::current_mpi_rank(), __FILE__, __LINE__,                  \
                         _s ? _s : "?");                                                       \
            std::abort();                                                                       \
        }                                                                                      \
    } while (0)
#endif

#ifndef ALPS_NCCLCHECK
#define ALPS_NCCLCHECK(cmd)                                                                    \
    do {                                                                                       \
        ncclResult_t _r = (cmd);                                                               \
        if (_r != ncclSuccess) {                                                               \
            std::fprintf(stderr, "[rank %d] NCCL error %s:%d '%s'\n",                           \
                         alps::detail::current_mpi_rank(), __FILE__, __LINE__,                  \
                         ncclGetErrorString(_r));                                              \
            std::abort();                                                                       \
        }                                                                                      \
    } while (0)
#endif

#ifndef ALPS_MPICHECK
#define ALPS_MPICHECK(cmd)                                                                     \
    do {                                                                                       \
        int _m = (cmd);                                                                        \
        if (_m != MPI_SUCCESS) {                                                               \
            char _s[MPI_MAX_ERROR_STRING];                                                     \
            int _l = 0;                                                                         \
            MPI_Error_string(_m, _s, &_l);                                                     \
            std::fprintf(stderr, "[rank %d] MPI error %s:%d '%s'\n",                            \
                         alps::detail::current_mpi_rank(), __FILE__, __LINE__, _s);             \
            std::abort();                                                                       \
        }                                                                                      \
    } while (0)
#endif
