// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once
// -----------------------------------------------------------------------------
// green_safe_elementwise.cuh -- plain-kernel elementwise transform/fill.
//
// thrust::transform (counting-iterator or binary) and large thrust::fill route
// to cub::DeviceTransform, which returns cudaErrorInvalidDevice on a CUDA
// green-context stream (the fused-transform launch path doesn't support green
// contexts; see [[green-context-cccl-incompat]]). A plain kernel launch runs
// correctly on green streams AND on every normal stream, so these hand-rolled
// helpers are drop-in green-safe replacements for the
//   thrust::transform(policy.on(stream), counting(0), counting(n), out, f)
//   thrust::fill(policy.on(stream), out, out+n, val)
// patterns that gamax operators run on the caller's stream. Default behaviour
// (a normal blocking / default stream) is byte-for-byte the same as before.
// -----------------------------------------------------------------------------
#include <cstdint>
#include <cuda_runtime.h>

namespace gamax::util::device {

template <typename OutT, typename F>
__global__ void indexed_transform_kernel(OutT* __restrict__ out,
                                         int64_t n, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f(i);
}

// out[i] = f(i) for i in [0, n). Green-safe replacement for
//   thrust::transform(policy.on(stream), counting(0), counting(n), out, f).
// `f` is an extended __device__ lambda taking an int64 index (matches the
// gamax::size_type the original lambdas used).
template <typename OutT, typename F>
inline void indexed_transform(OutT* out, int64_t n,
                              cudaStream_t stream, F f) {
    if (n <= 0) return;
    constexpr int block = 256;
    const unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    indexed_transform_kernel<OutT, F><<<grid, block, 0, stream>>>(out, n, f);
}

template <typename T>
__global__ void fill_value_kernel(T* __restrict__ out, T val, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = val;
}

// out[0..n) = val. Green-safe replacement for
//   thrust::fill(policy.on(stream), out, out+n, val).
template <typename T>
inline void device_fill(T* out, int64_t n, T val, cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int block = 256;
    const unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    fill_value_kernel<T><<<grid, block, 0, stream>>>(out, val, n);
}

} // namespace gamax::util::device
