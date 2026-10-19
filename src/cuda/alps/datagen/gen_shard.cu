// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/datagen/gen_shard.hpp"

#include <stdexcept>

#include "data_gen/table_gen.hpp"

namespace alps::datagen {

std::pair<std::unique_ptr<gamax::table>, std::unique_ptr<gamax::table>>
gen_shard(int rank, int size, const GenParams& p, cudaStream_t stream) {
    const gamax::size_type M   = p.pk_rows_per_rank;
    const gamax::size_type fkN = p.fk_rows_per_rank;
    if (M <= 0 || fkN <= 0) {
        throw std::runtime_error("alps::datagen::gen_shard: pk/fk rows must be positive");
    }
    // Exact r = fkN / M requires divisibility (own-block circular tiling).
    if (fkN % M != 0) {
        throw std::runtime_error(
            "alps::datagen::gen_shard: fk_rows_per_rank must be a multiple of pk_rows_per_rank");
    }

    // Disjoint contiguous PK block for this rank: [rank*M, (rank+1)*M).
    // Union over ranks = [0, size*M) -> globally unique & continuous.
    const uint64_t start = static_cast<uint64_t>(rank) * static_cast<uint64_t>(M);

    // PK: the block, each key once, shuffled order.
    auto pk = gamax::datagen::device::pk_uniform_id(
        p.key_type, M, p.ncols, p.seed + rank, stream, start);

    // FK: tile this rank's OWN block up to fkN rows -> each of its M keys
    // repeated exactly r = fkN/M times, shuffled. Union over ranks tiles the
    // global PK range exactly r times; hashing scatters each block uniformly,
    // so the shuffle exchange is balanced. match_ratio=1 -> all FK match.
    auto fk = gamax::datagen::device::fk_uniform_id(
        p.key_type, /*n_unique_keys=*/M, /*card=*/fkN, p.ncols,
        p.match_ratio, p.seed + 1000 + rank, stream, start);

    return {std::move(pk), std::move(fk)};
}

} // namespace alps::datagen
