// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "tpch/mpi_dist_helpers.hpp"

#include <limits>
#include <stdexcept>

#include "alps/context.hpp"
#include "alps/error.hpp"
#include "memory/row_utils.cuh"
#include "types/type_dispatcher.hpp"

namespace gamax::tpch::dist {

namespace {

__global__ void composite_hash_kernel(
    gamax::experimental::table_device_view keys,
    int64_t* out,
    gamax::size_type n)
{
    const auto i = static_cast<gamax::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    std::size_t hash = 0;
    for (gamax::size_type c = 0; c < keys.num_columns(); ++c) {
        auto col = keys.get_column(c);
        const std::size_t col_hash =
            gamax::type_dispatcher_all(
                col.type(), gamax::element_hash_functor{}, col, i);
        hash ^= col_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    out[i] = static_cast<int64_t>(hash);
}

bool fixed_width(gamax::data_type type) {
    return type.id() != gamax::type_id::STRING &&
           type.id() != gamax::type_id::EMPTY;
}

} // namespace

std::unique_ptr<gamax::column> make_composite_hash_column(
    gamax::table_view keys,
    cudaStream_t stream)
{
    if (keys.num_columns() == 0) {
        throw std::runtime_error("make_composite_hash_column requires at least one key column");
    }
    for (gamax::size_type c = 0; c < keys.num_columns(); ++c) {
        if (!fixed_width(keys[c].type())) {
            throw std::runtime_error(
                "make_composite_hash_column currently supports only fixed-width keys");
        }
    }

    auto out = gamax::make_column<int64_t>(keys.num_rows(), stream);
    if (keys.num_rows() == 0) return out;

    auto keys_dev = keys.device_view(stream);
    constexpr int threads = 256;
    const auto blocks = (keys.num_rows() + threads - 1) / threads;
    composite_hash_kernel<<<blocks, threads, 0, stream>>>(
        *keys_dev, out->mutable_view().data<int64_t>(), keys.num_rows());
    return out;
}

std::vector<double> sum_doubles(
    const std::vector<double>& local,
    const alps::Context& ctx)
{
    if (local.empty()) return {};
    if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("sum_doubles input is too large for MPI count");
    }

    std::vector<double> global(local.size(), 0.0);
    ALPS_MPICHECK(MPI_Allreduce(
        local.data(),
        global.data(),
        static_cast<int>(local.size()),
        MPI_DOUBLE,
        MPI_SUM,
        ctx.mpi_comm));
    return global;
}

} // namespace gamax::tpch::dist
