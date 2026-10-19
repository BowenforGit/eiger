// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cstddef>
#include <cstdint>

namespace alps::collectives {

// Hash-partition scheme identity. Every distributed hash exchange must use the
// same version/seed because the planner records only the partition keys, not a
// per-step seed. Keep this stable unless the scheme is deliberately versioned.
inline constexpr std::uint32_t kShuffleHashVersion = 2;
inline constexpr std::uint32_t kShuffleHashSeed = 0x51f15e5dU;
inline constexpr char kShuffleHashName[] = "xxh32-seeded-v2";
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));

#if defined(__CUDACC__)
#define ALPS_HOST_DEVICE __host__ __device__
#else
#define ALPS_HOST_DEVICE
#endif

ALPS_HOST_DEVICE inline std::uint32_t rotate_left(std::uint32_t value,
                                                  int shift) noexcept
{
    return (value << shift) | (value >> (32 - shift));
}

// Apply an independent avalanche before reducing a row hash to a rank. This
// is the exact four-byte XXH32 path used by cuco::xxhash_32<uint32_t>, kept
// self-contained so this public header remains usable by ordinary C++ files.
ALPS_HOST_DEVICE inline std::uint32_t shuffle_partition_hash(
    std::size_t row_hash) noexcept
{
    constexpr std::uint32_t prime2 = 0x85ebca77U;
    constexpr std::uint32_t prime3 = 0xc2b2ae3dU;
    constexpr std::uint32_t prime4 = 0x27d4eb2fU;
    constexpr std::uint32_t prime5 = 0x165667b1U;

    const auto folded = static_cast<std::uint32_t>(row_hash) ^
                        static_cast<std::uint32_t>(row_hash >> 32);
    std::uint32_t hash =
        kShuffleHashSeed + prime5 + static_cast<std::uint32_t>(sizeof(folded));
    hash += folded * prime3;
    hash = rotate_left(hash, 17) * prime4;
    hash ^= hash >> 15;
    hash *= prime2;
    hash ^= hash >> 13;
    hash *= prime3;
    hash ^= hash >> 16;
    return hash;
}

// Multiply-high reduction is unbiased for arbitrary positive world sizes.
ALPS_HOST_DEVICE inline int destination_from_row_hash(
    std::size_t row_hash, int world) noexcept
{
    const auto hash = shuffle_partition_hash(row_hash);
    return static_cast<int>(
        (static_cast<std::uint64_t>(hash) *
         static_cast<std::uint64_t>(world)) >> 32);
}

#undef ALPS_HOST_DEVICE

}  // namespace alps::collectives
