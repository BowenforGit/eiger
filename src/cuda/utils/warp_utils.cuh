#pragma once

#include <type_traits>

namespace gamax {
namespace warp {

// Returns a mask with bits set for lanes greater than the current lane.
// Accepts the runtime warp size to support both CUDA (32) and HIP (64) wavefronts.
template<typename MaskT>
__device__ __forceinline__ MaskT upper_lane_mask(int lane, int warp_size)
{
    if (lane + 1 >= warp_size) return MaskT{0};
    MaskT lower_bits = (MaskT{1} << (lane + 1)) - MaskT{1};
    return ~lower_bits;
}

// Popcount helper that matches the mask width (32-bit or 64-bit ballots).
template<typename MaskT>
__device__ __forceinline__ int popc_mask(MaskT mask)
{
    if constexpr (sizeof(MaskT) == sizeof(unsigned long long)) {
        return __popcll(static_cast<unsigned long long>(mask));
    } else {
        return __popc(static_cast<unsigned int>(mask));
    }
}

} // namespace warp
} // namespace gamax
