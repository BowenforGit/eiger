// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cuda_runtime_api.h>

#include "stats/stats_storage.hpp"
#include "memory/column.hpp"

namespace gamax::stats {

/**
 * @brief Interface for computing statistics on columns.
 * 
 * This file declares the functions that compute various statistics.
 * The actual implementations should be added in stats_compute.cu
 */

/**
 * @brief Compute requested statistics for a column and cache them.
 * 
 * This function checks if the requested statistics are already cached.
 * If not, it computes them and stores them in the StatsStorage.
 * 
 * @param column_view The column to compute statistics for
 * @param request The statistics request specifying which stats to compute
 * @param stream CUDA stream for async execution
 */
void compute_stats(
    gamax::column_view column,
    const StatsRequest& request,
    cudaStream_t stream);

/**
 * @brief Compute and return the minimum value of a column.
 * 
 * @param column The column to compute min for
 * @param stream CUDA stream for async execution
 * @return The minimum value as a scalar
 */
gamax::scalar compute_min(
    gamax::column_view column,
    cudaStream_t stream);

/**
 * @brief Compute and return the maximum value of a column.
 * 
 * @param column The column to compute max for
 * @param stream CUDA stream for async execution
 * @return The maximum value as a scalar
 */
gamax::scalar compute_max(
    gamax::column_view column,
    cudaStream_t stream);

/**
 * @brief Compute and return the average value of a column.
 * 
 * @param column The column to compute avg for
 * @param stream CUDA stream for async execution
 * @return The average value as a scalar (always FLOAT64)
 */
gamax::scalar compute_avg(
    gamax::column_view column,
    cudaStream_t stream);

/**
 * @brief Compute approximate NDV using HLL++.
 * 
 * @param column The column to compute NDV for
 * @param stream CUDA stream for async execution
 * @return Approximate number of distinct values as uint64_t scalar
 */
gamax::scalar compute_ndv_approx(
    gamax::column_view column,
    cudaStream_t stream);

/**
 * @brief Compute approximate inversion count.
 * 
 * @param column The column to compute inversion count for
 * @param stream CUDA stream for async execution
 * @return Approximate inversion count as uint64_t scalar
 */
gamax::scalar compute_inversion_count_approx(
    gamax::column_view column,
    cudaStream_t stream);

/**
 * @brief Compute equi-width histogram.
 * 
 * @param column The column to compute histogram for
 * @param num_buckets Number of buckets for the histogram
 * @param stream CUDA stream for async execution
 * @return Histogram with equi-width buckets
 */
std::shared_ptr<Histogram> compute_equi_width_histogram(
    gamax::column_view column,
    gamax::size_type num_buckets,
    cudaStream_t stream);

/**
 * @brief Compute equi-depth histogram.
 * 
 * @param column The column to compute histogram for
 * @param num_buckets Number of buckets for the histogram
 * @param stream CUDA stream for async execution
 * @return Histogram with approximately equal counts per bucket
 */
std::shared_ptr<Histogram> compute_equi_depth_histogram(
    gamax::column_view column,
    gamax::size_type num_buckets,
    cudaStream_t stream);

/**
 * @brief Helper to get or compute a scalar statistic.
 * 
 * Checks the cache first; if not found, computes and caches the result.
 * 
 * @param column The column data (also used as identifier via its data pointer)
 * @param type The statistic type (must be a scalar-returning stat)
 * @param stream CUDA stream for async execution
 * @return The scalar statistic value
 */
gamax::scalar get_or_compute_scalar_stat(
    gamax::column_view column,
    StatsType type,
    cudaStream_t stream);

/**
 * @brief Helper to get or compute a histogram statistic.
 * 
 * Checks the cache first; if not found, computes and caches the result.
 * 
 * @param column The column data (also used as identifier via its data pointer)
 * @param type The statistic type (must be histogram type)
 * @param num_buckets Number of buckets for histogram
 * @param stream CUDA stream for async execution
 * @return The histogram
 */
std::shared_ptr<Histogram> get_or_compute_histogram_stat(
    gamax::column_view column,
    StatsType type,
    cudaStream_t stream,
    gamax::size_type num_buckets = 100);

} // namespace gamax::stats
