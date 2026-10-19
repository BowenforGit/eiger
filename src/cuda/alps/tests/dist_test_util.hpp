// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Shared helpers for the alps MPI correctness tests (dist_join_test,
// shared_map_join_test): pick the MPI datatype for an element type, build a
// 2-column integer table from host vectors, copy a column back to host, and
// concatenate each rank's vector onto rank 0. Templated on the element type
// (int32_t / int64_t) so int32 and int64 tests share one implementation.

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"

namespace alps::testutil {

template <typename T> inline MPI_Datatype mpi_type();
template <> inline MPI_Datatype mpi_type<int32_t>() { return MPI_INT; }
template <> inline MPI_Datatype mpi_type<int64_t>() { return MPI_LONG_LONG; }

/// 2-column table [key | payload] from host vectors (device-resident on return).
template <typename T>
inline std::unique_ptr<gamax::table> make_table(const std::vector<T>& key,
                                                const std::vector<T>& payload,
                                                cudaStream_t stream) {
    std::vector<std::unique_ptr<gamax::column>> cols;
    cols.push_back(gamax::make_column_from_host(const_cast<T*>(key.data()),
                                                static_cast<gamax::size_type>(key.size()), stream));
    cols.push_back(gamax::make_column_from_host(const_cast<T*>(payload.data()),
                                                static_cast<gamax::size_type>(payload.size()), stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(cols));
}

/// Copy a device column's logical rows to a host vector.
template <typename T>
inline std::vector<T> col_to_host(const gamax::column& c, cudaStream_t stream) {
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    std::vector<T> h(c.size());
    if (c.size() > 0)
        ALPS_CUDACHECK(cudaMemcpy(h.data(), c.data<T>(), c.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

/// Concatenate each rank's vector onto rank 0 (empty on other ranks).
template <typename T>
inline std::vector<T> gather_to_root(const std::vector<T>& local, const alps::Context& ctx) {
    int n = static_cast<int>(local.size());
    std::vector<int> counts(ctx.size), displs(ctx.size);
    ALPS_MPICHECK(MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, ctx.mpi_comm));
    std::vector<T> out;
    int total = 0;
    if (ctx.rank == 0) {
        for (int r = 0; r < ctx.size; ++r) { displs[r] = total; total += counts[r]; }
        out.resize(total);
    }
    ALPS_MPICHECK(MPI_Gatherv(local.data(), n, mpi_type<T>(),
                              out.data(), counts.data(), displs.data(), mpi_type<T>(), 0, ctx.mpi_comm));
    return out;
}

} // namespace alps::testutil
