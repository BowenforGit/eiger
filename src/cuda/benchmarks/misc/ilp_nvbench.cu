// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

/**
 * @file ilp_nvbench.cu
 * @brief NVBench benchmark to explore L2 cache in-flight request limits.
 *
 * Measures throughput of L1-bypassing memory operations (using st.cg / ld.cg)
 * with varying instruction-level parallelism (VT = elements per thread).
 * All data fits in L2 cache so we isolate the L2 request handling capacity.
 *
 * The key idea: each thread issues VT independent memory requests via an
 * unrolled loop with strided access. As VT increases, throughput rises until
 * the per-warp/per-SM MSHR (Miss Status Holding Register) limit is hit,
 * revealing the maximum in-flight requests the hardware can sustain.
 *
 * Two benchmark variants:
 *   - bench_ilp_read:  Primary. Reads create trackable in-flight requests;
 *                       the warp stalls when MSHRs are exhausted.
 *   - bench_ilp_write: Supplementary. Stores are fire-and-forget; this
 *                       measures write buffer capacity instead.
 *
 * Axes: num_items (power-of-two), grid_dim, block_dim, vt
 *
 * Cache modifiers:
 *   - STORE_CG / LOAD_CG = bypass L1, cache at L2 (what we want)
 *   - STORE_CS / LOAD_CS = streaming, may evict from L2 prematurely (avoid)
 */

#include <nvbench/nvbench.cuh>

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cuda/std/__cccl/compiler.h>

#include <cmath>
#include <iostream>

#include "memory/mem_manager.hpp"
#include "memory/device_buffer.hpp"
#include "utils/launch.cuh"

constexpr int REPETITIONS = 1024;
constexpr bool synchronize = false;

// ============================================================================
// Kernels
// ============================================================================

/**
 * Write kernel: each thread performs VT strided stores bypassing L1.
 * Strided layout ensures each store hits a different cache line,
 * creating truly independent requests.
 */
template <int VT, typename T>
__global__ void write_kernel(T* __restrict__ output,
                             int n) {
    const int stride = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int wave = VT * stride;
    const int full_end = n - wave + 1; // last base where all VT accesses are in-bounds

    int base = tid;
    // Full waves: no bounds checking needed
    for (; base < full_end; base += wave) {
        T* __restrict__ base_ptr = output + base;
        _CCCL_PRAGMA_UNROLL(VT)
        for (int i = 0; i < VT; ++i) {
            cub::ThreadStore<cub::STORE_CG>(base_ptr + i * stride, static_cast<T>(base + i * stride));
        }
    }
    // Partial wave: bounds check required
    if (base < n) {
        T* __restrict__ base_ptr = output + base;
        _CCCL_PRAGMA_UNROLL(VT)
        for (int i = 0; i < VT; ++i) {
            const int idx = base + i * stride;
            if (idx < n) {
                cub::ThreadStore<cub::STORE_CG>(base_ptr + i * stride, static_cast<T>(idx));
            }
        }
    }
}

/**
 * Read kernel: each thread performs VT strided loads bypassing L1.
 * Values are XOR-accumulated to prevent dead-code elimination.
 * The final accumulated value is written to a single output location.
 */
template <int VT, typename T>
__global__ void read_kernel(T* input,
                            T* output,
                            int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile = blockDim.x * VT;
    const int stride = gridDim.x * tile;

    T acc = 0;

    for(int r = 0; r < REPETITIONS; ++r) {
        for (int blockOffset = blockIdx.x * tile; blockOffset < n; blockOffset += stride) {
            const T* base_ptr = input + blockOffset + threadIdx.x;
            
            _CCCL_PRAGMA_UNROLL(VT)
            for (int i = 0; i < VT; ++i) {
                acc += cub::ThreadLoad<cub::LOAD_CG>(base_ptr + i * blockDim.x);
            }

            if constexpr(synchronize) {
                __syncthreads(); // ensure all threads have issued their loads before next iteration
            }
        }
    }

    // Write accumulator to output to prevent dead-code elimination
    output[tid] = acc;
}

// ============================================================================
// VT compile-time dispatch
// ============================================================================

template <typename T>
void dispatch_write(int vt, dim3 grid, dim3 block, cudaStream_t stream,
                    T* output, int n) {
    switch (vt) {
        case 1:  write_kernel<1,  T><<<grid, block, 0, stream>>>(output, n); break;
        case 2:  write_kernel<2,  T><<<grid, block, 0, stream>>>(output, n); break;
        case 4:  write_kernel<4,  T><<<grid, block, 0, stream>>>(output, n); break;
        case 8:  write_kernel<8,  T><<<grid, block, 0, stream>>>(output, n); break;
        case 12: write_kernel<12, T><<<grid, block, 0, stream>>>(output, n); break;
        case 16: write_kernel<16, T><<<grid, block, 0, stream>>>(output, n); break;
        case 20: write_kernel<20, T><<<grid, block, 0, stream>>>(output, n); break;
        case 24: write_kernel<24, T><<<grid, block, 0, stream>>>(output, n); break;
        case 28: write_kernel<28, T><<<grid, block, 0, stream>>>(output, n); break;
        case 32: write_kernel<32, T><<<grid, block, 0, stream>>>(output, n); break;
        default: break;
    }
}

template <typename T>
void dispatch_read(int vt, dim3 grid, dim3 block, cudaStream_t stream,
                   T* input, T* output, int n) {
    switch (vt) {
        case 1:  read_kernel<1,  T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 2:  read_kernel<2,  T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 4:  read_kernel<4,  T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 8:  read_kernel<8,  T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 12: read_kernel<12, T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 16: read_kernel<16, T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 20: read_kernel<20, T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 24: read_kernel<24, T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 28: read_kernel<28, T><<<grid, block, 0, stream>>>(input, output, n); break;
        case 32: read_kernel<32, T><<<grid, block, 0, stream>>>(input, output, n); break;
        default: break;
    }
}

// ============================================================================
// Occupancy helpers
// ============================================================================

template <typename KernelFunc>
int get_max_blocks_per_sm(KernelFunc kernel, int block_size, size_t dynamic_smem = 0) {
    int max_blocks = 0;
    GAMAX_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MP(&max_blocks, kernel, block_size, dynamic_smem);
    return max_blocks;
}

template <typename T>
int dispatch_occupancy_read(int vt, int block_size) {
    switch (vt) {
        case 1:  return get_max_blocks_per_sm(read_kernel<1,  T>, block_size);
        case 2:  return get_max_blocks_per_sm(read_kernel<2,  T>, block_size);
        case 4:  return get_max_blocks_per_sm(read_kernel<4,  T>, block_size);
        case 8:  return get_max_blocks_per_sm(read_kernel<8,  T>, block_size);
        case 12: return get_max_blocks_per_sm(read_kernel<12, T>, block_size);
        case 16: return get_max_blocks_per_sm(read_kernel<16, T>, block_size);
        case 20: return get_max_blocks_per_sm(read_kernel<20, T>, block_size);
        case 24: return get_max_blocks_per_sm(read_kernel<24, T>, block_size);
        case 28: return get_max_blocks_per_sm(read_kernel<28, T>, block_size);
        case 32: return get_max_blocks_per_sm(read_kernel<32, T>, block_size);
        default: return 0;
    }
}

// ============================================================================
// Benchmark: ILP Write (supplementary)
// ============================================================================

template <typename T>
void bench_ilp_write(nvbench::state& state, nvbench::type_list<T>) {
    gamax::prewarm_device_memory_resource();

    const int num_items  = static_cast<int>(state.get_int64("num_items"));
    const int grid_dim    = static_cast<int>(state.get_int64("grid_dim"));
    const int block_dim   = static_cast<int>(state.get_int64("block_dim"));
    const int vt          = static_cast<int>(state.get_int64("vt"));

    // Skip if total threads * VT exceeds num_items (inner loop would never execute)
    if (static_cast<int64_t>(vt) * grid_dim * block_dim > num_items) {
        state.skip("VT * grid_dim * block_dim exceeds num_items");
        return;
    }

    // Verify data fits in L2 cache
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (static_cast<size_t>(num_items) * sizeof(T) > static_cast<size_t>(prop.l2CacheSize)) {
        state.skip("Data size exceeds L2 cache size");
        return;
    }

    // Allocate output buffer
    gamax::device_buffer output_buffer{
        static_cast<std::size_t>(num_items) * sizeof(T),
        cudaStreamDefault, gamax::buffer_init::zero};
    auto* d_output = static_cast<T*>(output_buffer.data());
    cudaDeviceSynchronize();

    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.add_element_count(num_items, "Elements");
    state.add_global_memory_writes<T>(num_items, "Writes");

    try {
        state.exec(nvbench::exec_tag::sync,
            [&](nvbench::launch& launch) {
                dispatch_write<T>(
                    vt,
                    dim3(static_cast<unsigned>(grid_dim)),
                    dim3(static_cast<unsigned>(block_dim)),
                    launch.get_stream(),
                    d_output, num_items);
                cudaStreamSynchronize(launch.get_stream());
            });
    } catch (const std::exception& e) {
        std::cerr << "ILP write benchmark error: " << e.what() << std::endl;
        state.skip(e.what());
    }

}

// ============================================================================
// Benchmark: ILP Read (primary)
// ============================================================================

template <typename T>
void bench_ilp_read(nvbench::state& state, nvbench::type_list<T>) {
    gamax::prewarm_device_memory_resource();

    const int num_items  = static_cast<int>(state.get_int64("num_items"));
    const int grid_dim    = static_cast<int>(state.get_int64("grid_dim"));
    const int block_dim   = static_cast<int>(state.get_int64("block_dim"));
    const int vt          = static_cast<int>(state.get_int64("vt"));
    const int num_threads = grid_dim * block_dim;

    // Skip if total threads * VT exceeds num_items
    if (static_cast<int64_t>(vt) * grid_dim * block_dim > num_items) {
        state.skip("VT * grid_dim * block_dim exceeds num_items");
        return;
    }

    int64_t stride_size = static_cast<int64_t>(vt) * grid_dim * block_dim;
    int64_t actual_num_items = ((num_items + stride_size - 1) / stride_size) * stride_size; // round up to nearest multiple of stride
    if (actual_num_items != num_items) {
        std::cout << "Warning: num_items (" << num_items << ") rounded up to " << actual_num_items
                  << " to ensure full waves of VT accesses." << std::endl;
    }


    // Verify data fits in L2 cache
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // if (static_cast<size_t>(actual_num_items) * sizeof(T) > static_cast<size_t>(prop.l2CacheSize)) {
    //     state.skip("Data size exceeds L2 cache size");
    //     return;
    // }

    // Allocate input (cleared = data brought into L2 via memset inside allocate_mem)
    gamax::device_buffer input_buffer{
        static_cast<std::size_t>(actual_num_items) * sizeof(T),
        cudaStreamDefault, gamax::buffer_init::zero};
    auto* d_input = static_cast<T*>(input_buffer.data());

    // Allocate output for accumulator write-back (one element per thread)
    gamax::device_buffer output_buffer{
        static_cast<std::size_t>(num_threads) * sizeof(T), cudaStreamDefault};
    auto* d_output = static_cast<T*>(output_buffer.data());
    cudaDeviceSynchronize();

    // Query occupancy
    const int max_blocks_per_sm = dispatch_occupancy_read<T>(vt, block_dim);
    const int num_sms = prop.multiProcessorCount;
    const int max_resident_blocks = max_blocks_per_sm * num_sms;
    const double num_waves = static_cast<double>(grid_dim) / max_resident_blocks;

    {
        auto& summary = state.add_summary("max_blocks_per_sm");
        summary.set_string("name", "max_blocks_per_sm");
        summary.set_string("description", "Max active thread blocks per SM");
        summary.set_int64("value", static_cast<int64_t>(max_blocks_per_sm));
    }
    {
        auto& summary = state.add_summary("num_waves");
        summary.set_string("name", "num_waves");
        summary.set_string("description", "Number of thread-block waves (grid_dim / (max_blocks_per_sm * num_SMs))");
        summary.set_float64("value", num_waves);
    }

    state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
    state.add_element_count(actual_num_items*REPETITIONS, "Elements");
    state.add_global_memory_reads<T>(actual_num_items*REPETITIONS, "Reads");

    try {
        state.exec([&](nvbench::launch& launch) {
                dispatch_read<T>(
                    vt,
                    dim3(static_cast<unsigned>(grid_dim)),
                    dim3(static_cast<unsigned>(block_dim)),
                    launch.get_stream(),
                    d_input, d_output, actual_num_items);
            });
    } catch (const std::exception& e) {
        std::cerr << "ILP read benchmark error: " << e.what() << std::endl;
        state.skip(e.what());
    }

}

// ============================================================================
// NVBench Registration
// ============================================================================

NVBENCH_BENCH_TYPES(bench_ilp_write,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t>))
.set_name("bench_ilp_write")
.set_type_axes_names({"ValueType"})
.add_int64_power_of_two_axis("num_items", {22})        // 256K, 1M, 4M elements
.add_int64_axis("grid_dim",  {1, 2, 4, 8, 16, 32, 64, 128, 256})
.add_int64_axis("block_dim", {1024})
.add_int64_axis("vt",        {1, 2, 4, 8, 12, 16, 20, 24, 28, 32});

NVBENCH_BENCH_TYPES(bench_ilp_read,
    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t>))
.set_name("bench_ilp_read")
.set_type_axes_names({"ValueType"})
// .add_int64_power_of_two_axis("num_items", {18, 20, 22})        // 256K, 1M, 4M elements
.add_int64_power_of_two_axis("num_items", {22})        // 256K, 1M, 4M elements
.add_int64_axis("grid_dim",  {1, 4, 16, 64, 256, 512, 1024})
.add_int64_axis("block_dim", {256})
// .add_int64_axis("block_dim", {32, 64, 128, 256, 512, 1024})
.add_int64_axis("vt",        {1, 2, 4, 8, 12, 16, 20});
