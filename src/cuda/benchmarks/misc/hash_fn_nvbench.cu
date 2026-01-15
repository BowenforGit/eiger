/**
 * @file hash_fn_nvbench.cu
 * @brief NVBench benchmark for GPU hash functions from cuco library.
 * 
 * Dimensions explored:
 * - Data types: int32_t, int64_t
 * - Number of values: Power of 2 scaling
 * - Hash functions: identity, murmurhash3_32, murmurhash3_fmix_32/64, xxhash_32/64
 * - Seed values
 * - Output mode: write output vs reduce (to isolate compute from memory bandwidth)
 */

#include <nvbench/nvbench.cuh>

#include <cub/cub.cuh>
#include <cuco/hash_functions.cuh>

#include <string>
#include <iostream>
#include <type_traits>

#include "data_gen/table_gen.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"
#include "atomics/atomics.cuh"

// ============================================================================
// Hash function type tags for compile-time dispatch
// ============================================================================

struct hash_identity {};
struct hash_murmurhash3_32 {};
struct hash_murmurhash3_fmix_32 {};
struct hash_murmurhash3_fmix_64 {};
struct hash_xxhash_32 {};
struct hash_xxhash_64 {};

// ============================================================================
// Hash function factory - creates the appropriate hash function for a type
// ============================================================================

template<typename KeyT, typename HashTag>
struct hash_factory {
    static constexpr bool is_supported = false;
};

// Identity hash - supported for all integral types
template<typename KeyT>
struct hash_factory<KeyT, hash_identity> {
    static constexpr bool is_supported = true;
    using hash_type = cuco::identity_hash<KeyT>;
    static hash_type create(uint64_t /*seed*/ = 0) { return hash_type{}; }
};

// MurmurHash3_32 - supported for all types
template<typename KeyT>
struct hash_factory<KeyT, hash_murmurhash3_32> {
    static constexpr bool is_supported = true;
    using hash_type = cuco::murmurhash3_32<KeyT>;
    static hash_type create(uint32_t seed = 0) { return hash_type{seed}; }
};

// MurmurHash3_fmix_32 - only for 4-byte types
template<typename KeyT>
struct hash_factory<KeyT, hash_murmurhash3_fmix_32> {
    static constexpr bool is_supported = (sizeof(KeyT) == 4);
    using hash_type = cuco::murmurhash3_fmix_32<KeyT>;
    static hash_type create(uint32_t seed = 0) { return hash_type{seed}; }
};

// MurmurHash3_fmix_64 - only for 8-byte types
template<typename KeyT>
struct hash_factory<KeyT, hash_murmurhash3_fmix_64> {
    static constexpr bool is_supported = (sizeof(KeyT) == 8);
    using hash_type = cuco::murmurhash3_fmix_64<KeyT>;
    static hash_type create(uint64_t seed = 0) { return hash_type{seed}; }
};

// XXHash_32 - supported for all types
template<typename KeyT>
struct hash_factory<KeyT, hash_xxhash_32> {
    static constexpr bool is_supported = true;
    using hash_type = cuco::xxhash_32<KeyT>;
    static hash_type create(uint32_t seed = 0) { return hash_type{seed}; }
};

// XXHash_64 - supported for all types
template<typename KeyT>
struct hash_factory<KeyT, hash_xxhash_64> {
    static constexpr bool is_supported = true;
    using hash_type = cuco::xxhash_64<KeyT>;
    static hash_type create(uint64_t seed = 0) { return hash_type{seed}; }
};

// ============================================================================
// Kernel: Apply hash function and write output (read from memory)
// ============================================================================

template<typename KeyT, typename HashFn, typename OutputT>
__global__ void hash_kernel_write(
    const KeyT* __restrict__ input,
    OutputT* __restrict__ output,
    int64_t num_items,
    HashFn hash_fn)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    for (int64_t i = idx; i < num_items; i += stride) {
        output[i] = hash_fn(input[i]);
    }
}

// ============================================================================
// Kernel: Apply hash function and write output (no memory read, counting)
// ============================================================================

template<typename KeyT, typename HashFn, typename OutputT>
__global__ void hash_kernel_write_noread(
    OutputT* __restrict__ output,
    int64_t num_items,
    KeyT start_value,
    HashFn hash_fn)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    for (int64_t i = idx; i < num_items; i += stride) {
        // Generate value on-the-fly like thrust::counting_iterator
        KeyT value = start_value + static_cast<KeyT>(i);
        output[i] = hash_fn(value);
    }
}

// ============================================================================
// Kernel: Apply hash function and reduce using CUB BlockReduce + atomicAdd
// ============================================================================

template<int BlockSize, typename KeyT, typename HashFn, typename OutputT>
__global__ void hash_kernel_reduce(
    const KeyT* __restrict__ input,
    OutputT* __restrict__ global_result,
    int64_t num_items,
    HashFn hash_fn)
{
    using BlockReduceT = cub::BlockReduce<OutputT, BlockSize>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    // Each thread computes XOR of its assigned elements
    OutputT thread_result = 0;
    for (int64_t i = idx; i < num_items; i += stride) {
        thread_result ^= hash_fn(input[i]);
    }
    
    // Block-level reduction using CUB
    OutputT block_result = BlockReduceT(temp_storage).Sum(thread_result);
    
    // Atomically update global counter
    if (threadIdx.x == 0) {
        gamax::atomic::atomic_add(global_result, block_result);
    }
}

// ============================================================================
// Kernel: Apply hash function and reduce (no memory read, counting)
// ============================================================================

template<int BlockSize, typename KeyT, typename HashFn, typename OutputT>
__global__ void hash_kernel_reduce_noread(
    OutputT* __restrict__ global_result,
    int64_t num_items,
    KeyT start_value,
    HashFn hash_fn)
{
    using BlockReduceT = cub::BlockReduce<OutputT, BlockSize>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    // Each thread computes XOR of its assigned elements
    OutputT thread_result = 0;
    for (int64_t i = idx; i < num_items; i += stride) {
        // Generate value on-the-fly like thrust::counting_iterator
        KeyT value = start_value + static_cast<KeyT>(i);
        thread_result ^= hash_fn(value);
    }
    
    // Block-level reduction using CUB
    OutputT block_result = BlockReduceT(temp_storage).Sum(thread_result);
    
    // Atomically update global counter
    if (threadIdx.x == 0) {
        gamax::atomic::atomic_add(global_result, block_result);
    }
}

// ============================================================================
// Main benchmark
// ============================================================================

template<typename KeyT, typename HashTag>
void bench_hash_throughput(nvbench::state& state, nvbench::type_list<KeyT, HashTag>) {
    release_all_mem();
    
    using Factory = hash_factory<KeyT, HashTag>;
    
    // Check if this hash function is supported for this type
    if constexpr (!Factory::is_supported) {
        state.skip("Hash function not supported for this data type");
        return;
    } else {
        // Get parameters
        int64_t num_items = state.get_int64("num_items");
        uint64_t seed = static_cast<uint64_t>(state.get_int64_or_default("seed", 0));
        bool use_reduce = state.get_int64_or_default("reduce", 0) != 0;
        bool read_input = state.get_int64_or_default("read_input", 1) != 0;
        
        // Create hash function
        auto hash_fn = Factory::create(seed);
        using OutputT = typename Factory::hash_type::result_type;
        
        // Generate input data only if we need to read from memory
        std::unique_ptr<gamax::table> input_table;
        const KeyT* d_input = nullptr;
        if (read_input) {
            auto key_type = gamax::data_type(gamax::type_to_id<KeyT>());
            input_table = gamax::datagen::device::pk_uniform_id(key_type, num_items, 1, 42);
            d_input = input_table->get_column(0).view().template data<KeyT>();
        }
        KeyT start_value = static_cast<KeyT>(0);  // For counting iterator mode
        
        // Allocate output
        constexpr int block_size = 256;
        int num_blocks = (num_items + block_size - 1) / block_size;
        
        gamax::column_ptr output_col;
        if (use_reduce) {
            // Single element for atomic accumulation
            output_col = gamax::make_column<OutputT>(1, true);
        } else {
            output_col = gamax::make_column<OutputT>(num_items);
        }
        OutputT* d_output = output_col->data<OutputT>();
        
        // Add throughput metrics
        state.add_element_count(num_items, "Elements");
        if (read_input) {
            state.add_global_memory_reads<KeyT>(num_items, "InputBytes");
        }
        if (!use_reduce) {
            state.add_global_memory_writes<OutputT>(num_items, "OutputBytes");
        }
        
        cudaDeviceSynchronize();
        state.set_cuda_stream(nvbench::make_cuda_stream_view(0));
        
        if (use_reduce && read_input) {
            state.exec(nvbench::exec_tag::sync,
                [&](nvbench::launch& launch) {
                    hash_kernel_reduce<block_size><<<num_blocks, block_size, 0, launch.get_stream()>>>(
                        d_input, d_output, num_items, hash_fn);
                    cudaStreamSynchronize(launch.get_stream());
                });
        } else if (use_reduce && !read_input) {
            state.exec(nvbench::exec_tag::sync,
                [&](nvbench::launch& launch) {
                    hash_kernel_reduce_noread<block_size><<<num_blocks, block_size, 0, launch.get_stream()>>>(
                        d_output, num_items, start_value, hash_fn);
                    cudaStreamSynchronize(launch.get_stream());
                });
        } else if (!use_reduce && read_input) {
            state.exec(nvbench::exec_tag::sync,
                [&](nvbench::launch& launch) {
                    hash_kernel_write<<<num_blocks, block_size, 0, launch.get_stream()>>>(
                        d_input, d_output, num_items, hash_fn);
                    cudaStreamSynchronize(launch.get_stream());
                });
        } else {  // !use_reduce && !read_input
            state.exec(nvbench::exec_tag::sync,
                [&](nvbench::launch& launch) {
                    hash_kernel_write_noread<<<num_blocks, block_size, 0, launch.get_stream()>>>(
                        d_output, num_items, start_value, hash_fn);
                    cudaStreamSynchronize(launch.get_stream());
                });
        }
    }
}

// ============================================================================
// Type lists
// ============================================================================

using key_types = nvbench::type_list<int32_t, int64_t>;

using hash_functions = nvbench::type_list<
    hash_identity,
    hash_murmurhash3_32,
    hash_murmurhash3_fmix_32,
    hash_murmurhash3_fmix_64,
    hash_xxhash_32,
    hash_xxhash_64>;

// ============================================================================
// Register benchmark
// ============================================================================

NVBENCH_BENCH_TYPES(bench_hash_throughput,
    NVBENCH_TYPE_AXES(key_types, hash_functions))
.set_name("bench_hash_throughput")
.set_type_axes_names({"KeyType", "HashFunction"})
.add_int64_power_of_two_axis("num_items", nvbench::range(20, 30, 2))
.add_int64_axis("seed", {42})
.add_int64_axis("reduce", {0, 1})
.add_int64_axis("read_input", {0, 1});
