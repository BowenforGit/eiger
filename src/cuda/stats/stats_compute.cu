// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "stats/stats_compute.hpp"
#include "types/type_dispatcher.hpp"
#include "memory/mem_manager.hpp"
#include "string/string_column.hpp"
#include "string/string_impl.cuh"

#include <stdexcept>
#include <limits>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cuco/hyperloglog.cuh>
#include "utils/green_safe_elementwise.cuh"

namespace gamax::stats {

// ============================================================================
// Min/Max/Sum reduction functors for thrust::reduce
// ============================================================================

template<typename T>
struct MinOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return a < b ? a : b;
    }
};

template<typename T>
struct MaxOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return a > b ? a : b;
    }
};

template<typename T>
struct SumOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

// ============================================================================
// Type-dispatched implementations
// ============================================================================

struct compute_min_impl {
    template<typename T>
    gamax::scalar operator()(gamax::column_view column, cudaStream_t stream) const {
        auto result = thrust::reduce(
            gamax::rmm_sync_exec_policy_handle{stream}.get(),
            column.data<T>(),
            column.data<T>() + column.size(),
            std::numeric_limits<T>::max(),
            MinOp<T>{}
        );
        cudaStreamSynchronize(stream);
        return gamax::scalar(result);
    }
};

struct compute_max_impl {
    template<typename T>
    gamax::scalar operator()(gamax::column_view column, cudaStream_t stream) const {
        auto result = thrust::reduce(
            gamax::rmm_sync_exec_policy_handle{stream}.get(),
            column.data<T>(),
            column.data<T>() + column.size(),
            std::numeric_limits<T>::lowest(),
            MaxOp<T>{}
        );
        cudaStreamSynchronize(stream);
        return gamax::scalar(result);
    }
};

struct compute_sum_impl {
    template<typename T>
    auto operator()(gamax::column_view column, cudaStream_t stream) const {
        // Use 64-bit types for accumulation to avoid overflow
        // For floating point, use double; for signed integers, use int64_t; for unsigned, use uint64_t
        using AccumType = std::conditional_t<std::is_floating_point_v<T>, double,
                          std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>>;
        
        auto result = thrust::reduce(
            gamax::rmm_sync_exec_policy_handle{stream}.get(),
            column.data<T>(),
            column.data<T>() + column.size(),
            static_cast<AccumType>(0),
            thrust::plus<AccumType>{}
        );
        cudaStreamSynchronize(stream);
        return static_cast<double>(result);  // Return as double for avg calculation
    }
};

// Type alias for 4-byte string prefix used in string NDV estimation
using prefix_t = int32_t;
constexpr size_t PREFIX_SIZE = sizeof(prefix_t);

// Type alias for string column implementation (defined at namespace scope for NVCC compatibility)
using string_col_impl_t = gamax::detail::id_to_string_impl<gamax::ARROW_STRING>::string_col_impl;

struct compute_ndv_impl {
    // Numeric columns: use HyperLogLog directly on values
    template<typename T>
    gamax::size_type operator()(gamax::column_view column, cudaStream_t stream) const {
        if constexpr (!std::is_same_v<T, gamax::string_type_tag>) {
            auto const sd = cuco::standard_deviation{0.0122197};
            using allocator_type = gamax::rmm_cuco_allocator<cuda::std::byte>;
            gamax::rmm_allocator_handle<cuda::std::byte> allocator{stream};
            cuco::hyperloglog<
                T, cuda::thread_scope_device, cuco::xxhash_64<T>, allocator_type>
                estimator{sd, {}, allocator.get(), cuda::stream_ref{stream}};
            
            estimator.add(column.data<T>(), column.data<T>() + column.size(), stream);
            cudaStreamSynchronize(stream);

            // estimate() copies the HLL sketch device->host via
            // cudaMemcpyBatchAsync, which SEGFAULTS on a CUDA green-context
            // stream (the batched-memcpy driver path doesn't support green
            // contexts). The sketch is already complete (synced above), so
            // read it back on the legacy default stream — a non-green stream
            // where the batched copy works. Correct for every caller (this is
            // a host-returning estimate; the sketch build already synced).
            return static_cast<gamax::size_type>(
                estimator.estimate(cudaStream_t{0}));
        } else {
            gamax::string_column_view str_col(column, gamax::ARROW_STRING);
            const auto num_strings = str_col.size();
            
            // Allocate temporary buffer for prefixes
            auto prefix_col = gamax::make_column<prefix_t>(num_strings, stream);
            
            // Extract 4-byte prefixes from each string
            // Green-safe: thrust::transform over a counting range routes to
            // cub::DeviceTransform, which fails on green-context streams.
            gamax::util::device::indexed_transform(
                prefix_col->template data<prefix_t>(), num_strings, stream,
                [str_col] __device__ (gamax::size_type idx) {
                    prefix_t prefix = static_cast<prefix_t>(0);
                    string_col_impl_t str_impl(str_col);
                    auto str = str_impl.viewAt(idx);
                    auto len = str.size() < PREFIX_SIZE ? str.size() : PREFIX_SIZE;
                    const char* str_data = str.data();
                    for (size_t i = 0; i < len; ++i) {
                        reinterpret_cast<char*>(&prefix)[PREFIX_SIZE - 1 - i] = str_data[i];
                    }
                    return prefix;
                });
            
            // Use HyperLogLog on the extracted prefixes
            auto const sd = cuco::standard_deviation{0.0122197};
            using allocator_type = gamax::rmm_cuco_allocator<cuda::std::byte>;
            gamax::rmm_allocator_handle<cuda::std::byte> allocator{stream};
            cuco::hyperloglog<
                prefix_t,
                cuda::thread_scope_device,
                cuco::xxhash_64<prefix_t>,
                allocator_type>
                estimator{sd, {}, allocator.get(), cuda::stream_ref{stream}};
            
            estimator.add(
                prefix_col->template data<prefix_t>(),
                prefix_col->template data<prefix_t>() + num_strings,
                stream);
            cudaStreamSynchronize(stream);

            // Same green-safe reroute as the numeric branch: estimate()'s
            // cudaMemcpyBatchAsync segfaults on a green-context stream; the
            // sketch is complete (synced above), so read it on stream 0.
            return static_cast<gamax::size_type>(
                estimator.estimate(cudaStream_t{0}));
        }
    }
};

// ============================================================================
// Public API implementations
// ============================================================================

gamax::scalar compute_min(
    gamax::column_view column,
    cudaStream_t stream)
{
    return gamax::type_dispatcher(column.type(), compute_min_impl{}, column, stream);
}

gamax::scalar compute_max(
    gamax::column_view column,
    cudaStream_t stream)
{
    return gamax::type_dispatcher(column.type(), compute_max_impl{}, column, stream);
}

gamax::scalar compute_avg(
    gamax::column_view column,
    cudaStream_t stream)
{
    if (column.size() == 0) {
        return gamax::scalar(static_cast<double>(0));
    }
    
    double sum = gamax::type_dispatcher(column.type(), compute_sum_impl{}, column, stream);
    double avg = sum / static_cast<double>(column.size());
    return gamax::scalar(avg);
}

gamax::scalar compute_ndv_approx(
    gamax::column_view column,
    cudaStream_t stream)
{
    if (column.size() == 0) {
        return gamax::scalar(static_cast<gamax::size_type>(0));
    }
    
    gamax::size_type ndv = gamax::type_dispatcher_all(column.type(), compute_ndv_impl{}, column, stream);
    return gamax::scalar(ndv);
}

gamax::scalar compute_inversion_count_approx(
    gamax::column_view column,
    cudaStream_t stream)
{
    // TODO: Implement approximate inversion count
    throw std::runtime_error("compute_inversion_count_approx not yet implemented");
}

std::shared_ptr<Histogram> compute_equi_width_histogram(
    gamax::column_view column,
    gamax::size_type num_buckets,
    cudaStream_t stream)
{
    // TODO: Implement equi-width histogram computation
    throw std::runtime_error("compute_equi_width_histogram not yet implemented");
}

std::shared_ptr<Histogram> compute_equi_depth_histogram(
    gamax::column_view column,
    gamax::size_type num_buckets,
    cudaStream_t stream)
{
    // TODO: Implement equi-depth histogram computation
    throw std::runtime_error("compute_equi_depth_histogram not yet implemented");
}

// ============================================================================
// Helper functions for get-or-compute pattern
// ============================================================================

gamax::scalar get_or_compute_scalar_stat(
    gamax::column_view column,
    StatsType type,
    cudaStream_t stream)
{
    auto& storage = StatsStorage::instance();
    
    // Check cache first
    auto cached = storage.get_scalar_stat(column, type);
    if (cached) {
        return *cached;
    }
    
    // Compute the statistic
    gamax::scalar result;
    switch (type) {
        case StatsType::MIN:
            result = compute_min(column, stream);
            break;
        case StatsType::MAX:
            result = compute_max(column, stream);
            break;
        case StatsType::AVG:
            result = compute_avg(column, stream);
            break;
        case StatsType::NDV_APPROX:
            result = compute_ndv_approx(column, stream);
            break;
        case StatsType::INV_COUNT_APPROX:
            result = compute_inversion_count_approx(column, stream);
            break;
        default:
            throw std::runtime_error("Invalid scalar stat type requested");
    }
    
    // Cache the result
    storage.set_stat(column, type, StatsValue(result));
    
    return result;
}

std::shared_ptr<Histogram> get_or_compute_histogram_stat(
    gamax::column_view column,
    StatsType type,
    cudaStream_t stream,
    gamax::size_type num_buckets)
{
    auto& storage = StatsStorage::instance();
    
    // Check cache first
    auto cached = storage.get_histogram_stat(column, type);
    if (cached) {
        return cached;
    }
    
    // Compute the histogram
    std::shared_ptr<Histogram> result;
    switch (type) {
        case StatsType::EQUI_WIDTH_HISTOGRAM:
            result = compute_equi_width_histogram(column, num_buckets, stream);
            break;
        case StatsType::EQUI_DEPTH_HISTOGRAM:
            result = compute_equi_depth_histogram(column, num_buckets, stream);
            break;
        default:
            throw std::runtime_error("Invalid histogram stat type requested");
    }
    
    // Cache the result
    storage.set_stat(column, type, StatsValue(result));
    
    return result;
}

void compute_stats(
    gamax::column_view column,
    const StatsRequest& request,
    cudaStream_t stream)
{
    // Register the column with the specified is_base flag if not already registered
    auto& storage = StatsStorage::instance();
    if (!storage.is_registered(column)) {
        storage.register_column(column, request.is_base);
    }
    
    for (auto type : request.stats_types) {
        switch (type) {
            case StatsType::MIN:
            case StatsType::MAX:
            case StatsType::AVG:
            case StatsType::NDV_APPROX:
            case StatsType::INV_COUNT_APPROX:
                get_or_compute_scalar_stat(column, type, stream);
                break;
            case StatsType::EQUI_WIDTH_HISTOGRAM:
            case StatsType::EQUI_DEPTH_HISTOGRAM:
                get_or_compute_histogram_stat(column, type, 
                                              stream, request.num_buckets);
                break;
            default:
                // Skip unsupported types
                break;
        }
    }
}

} // namespace gamax::stats
