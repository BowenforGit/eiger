// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Shared hash + destination-rank mapping for the shuffle. Both the shuffle
// kernel and the tests use these so the tests check the *real* routing.

#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

namespace alps::collectives {

// Knuth multiplicative ("Fibonacci") hash -> 32-bit. Best-distributed bits land
// in the high end of the word, which is what the range-reduction below reads.
template <typename KeyT>
__host__ __device__ __forceinline__ int32_t hash_key(KeyT k) {
    using U = std::make_unsigned_t<KeyT>;
    uint64_t x = static_cast<uint64_t>(static_cast<U>(k));
    x ^= x >> 32;
    uint32_t h = static_cast<uint32_t>(x) * 2654435761u;
    h ^= h >> 15;
    return static_cast<int32_t>(h);
}

// Destination rank in [0, world) via multiplicative range reduction of the hash:
//   dest = (uint32(h) * world) >> 32.
// This maps the 32-bit hash uniformly onto `world` ranks for ANY world (no
// modulo bias toward the low ranks). For a power-of-two world = 2^b it reduces
// exactly to (uint32(h) >> (32 - b)) — the top b bits — so power-of-two routing
// is byte-for-byte unchanged. The shuffle then radix-partitions rows by the low
// ceil(log2(world)) bits of this small, non-negative [0, world) value, which is
// immune to the sign-bit twiddling CUB applies to signed keys.
__host__ __device__ __forceinline__ int dest_rank_from_hash(int32_t h, int world) {
    const uint32_t uh = static_cast<uint32_t>(h);
    return static_cast<int>((static_cast<uint64_t>(uh) * static_cast<uint64_t>(world)) >> 32);
}

template <typename KeyT>
__host__ __device__ __forceinline__ int dest_rank(KeyT k, int world) {
    return dest_rank_from_hash(hash_key<KeyT>(k), world);
}

} // namespace alps::collectives
