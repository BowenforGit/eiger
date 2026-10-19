// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <unordered_map>
#include <memory>
#include <variant>
#include <optional>
#include <vector>
#include <mutex>
#include <functional>
#include <atomic>
#include <string>

#include "memory/scalar.hpp"
#include "memory/column.hpp"

namespace gamax::stats {

/**
 * @brief Enumeration of supported statistics types.
 */
enum class StatsType {
    MIN,
    MAX,
    AVG,
    NDV_APPROX,             // HLL++ based NDV approximation
    INV_COUNT_APPROX,       // approximate inversion count
    EQUI_WIDTH_HISTOGRAM,
    EQUI_DEPTH_HISTOGRAM,
    // Add more stats types as needed
    UNSUPPORTED_STATS_TYPE
};

/**
 * @brief Returns a string representation of the StatsType.
 */
inline const char* stats_type_to_string(StatsType type) {
    switch (type) {
        case StatsType::MIN: return "MIN";
        case StatsType::MAX: return "MAX";
        case StatsType::AVG: return "AVG";
        case StatsType::NDV_APPROX: return "NDV_APPROX";
        case StatsType::INV_COUNT_APPROX: return "INV_COUNT_APPROX";
        case StatsType::EQUI_WIDTH_HISTOGRAM: return "EQUI_WIDTH_HISTOGRAM";
        case StatsType::EQUI_DEPTH_HISTOGRAM: return "EQUI_DEPTH_HISTOGRAM";
        default: return "UNSUPPORTED_STATS_TYPE";
    }
}



/**
 * @brief Histogram data structure for equi-width or equi-depth histograms.
 * 
 * Contains bucket boundaries and counts.
 */
struct Histogram {
    std::unique_ptr<gamax::column> boundaries;  // Bucket boundaries (n+1 values for n buckets)
    std::unique_ptr<gamax::column> counts;      // Count per bucket
    gamax::size_type num_buckets{0};
    
    Histogram() = default;
    Histogram(Histogram&&) = default;
    Histogram& operator=(Histogram&&) = default;
    
    // Deep copy for histogram
    Histogram deep_copy(cudaStream_t stream) const {
        Histogram copy;
        if (boundaries) {
            copy.boundaries = boundaries->deep_copy(stream);
        }
        if (counts) {
            copy.counts = counts->deep_copy(stream);
        }
        copy.num_buckets = num_buckets;
        return copy;
    }
};

/**
 * @brief A variant type that can hold different statistics values.
 * 
 * - gamax::scalar: For scalar statistics like MIN, MAX, AVG, NDV_APPROX
 * - std::shared_ptr<Histogram>: For histogram-based statistics
 * - std::shared_ptr<gamax::column>: For column-based statistics (e.g., HLL sketch)
 */
using StatsValue = std::variant<
    gamax::scalar,                      // For scalar stats (MIN, MAX, AVG, NDV, etc.)
    std::shared_ptr<Histogram>,         // For histogram stats
    std::shared_ptr<gamax::column>      // For column-based stats (e.g., HLL registers)
>;

/**
 * @brief Storage for statistics of a single column.
 */
class ColumnStats {
public:
    ColumnStats() = default;
    explicit ColumnStats(bool is_base) : _is_base(is_base) {}
    
    /**
     * @brief Check if a specific statistic type has been computed.
     */
    bool has_stat(StatsType type) const {
        return _stats.find(type) != _stats.end();
    }
    
    /**
     * @brief Get a statistic value. Returns std::nullopt if not computed.
     */
    std::optional<StatsValue> get_stat(StatsType type) const {
        auto it = _stats.find(type);
        if (it != _stats.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    /**
     * @brief Get a scalar statistic. Returns std::nullopt if not computed or not a scalar.
     */
    std::optional<gamax::scalar> get_scalar_stat(StatsType type) const {
        auto stat = get_stat(type);
        if (stat && std::holds_alternative<gamax::scalar>(*stat)) {
            return std::get<gamax::scalar>(*stat);
        }
        return std::nullopt;
    }
    
    /**
     * @brief Get a histogram statistic. Returns nullptr if not computed or not a histogram.
     */
    std::shared_ptr<Histogram> get_histogram_stat(StatsType type) const {
        auto stat = get_stat(type);
        if (stat && std::holds_alternative<std::shared_ptr<Histogram>>(*stat)) {
            return std::get<std::shared_ptr<Histogram>>(*stat);
        }
        return nullptr;
    }
    
    /**
     * @brief Store a statistic value.
     */
    void set_stat(StatsType type, StatsValue value) {
        _stats[type] = std::move(value);
    }
    
    /**
     * @brief Remove a specific statistic.
     */
    void remove_stat(StatsType type) {
        _stats.erase(type);
    }
    
    /**
     * @brief Clear all statistics for this column.
     */
    void clear() {
        _stats.clear();
    }
    
    /**
     * @brief Check if this is a base table column.
     */
    bool is_base() const { return _is_base; }
    
    /**
     * @brief Get all computed stats types.
     */
    std::vector<StatsType> computed_stats() const {
        std::vector<StatsType> result;
        result.reserve(_stats.size());
        for (const auto& [type, _] : _stats) {
            result.push_back(type);
        }
        return result;
    }

private:
    bool _is_base{false};
    std::unordered_map<StatsType, StatsValue> _stats;
};

/**
 * @brief Central storage for column statistics.
 * 
 * This class provides thread-safe access to column statistics and
 * supports distinguishing between base and intermediate columns.
 * 
 * Columns are identified by a monotonic logical owner token plus subview
 * offset/size/type. Device addresses are deliberately excluded because an
 * allocator can reuse an address for unrelated column contents.
 */
class StatsStorage {
public:
    struct ColumnCacheKey {
        gamax::column_identity identity{};
        gamax::size_type offset{};
        gamax::size_type size{};
        gamax::type_id type{gamax::type_id::EMPTY};

        bool operator==(ColumnCacheKey const& other) const noexcept {
            return identity == other.identity && offset == other.offset &&
                   size == other.size && type == other.type;
        }
    };

    struct ColumnCacheKeyHash {
        std::size_t operator()(ColumnCacheKey const& key) const noexcept {
            auto mix = [](std::size_t seed, std::size_t value) {
                return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
            };
            std::size_t result = std::hash<gamax::column_identity>{}(key.identity);
            result = mix(result, std::hash<gamax::size_type>{}(key.offset));
            result = mix(result, std::hash<gamax::size_type>{}(key.size));
            return mix(result, std::hash<int>{}(static_cast<int>(key.type)));
        }
    };

    /**
     * @brief Get the singleton instance of the stats storage.
     */
    static StatsStorage& instance() {
        // Explicit reset/shutdown clears GPU owners. Leak the host registry so
        // static destruction can never invoke CUDA after runtime teardown.
        static auto* storage = new StatsStorage;
        return *storage;
    }
    
    /**
     * @brief Register a column for statistics tracking.
     * 
     * @param column The identified logical column/subview
     * @param is_base True if this is a base table column, false if intermediate
     */
    void register_column(gamax::column_view column, bool is_base = false) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto const key = cache_key(column);
        if (_column_stats.find(key) == _column_stats.end()) {
            _column_stats[key] = ColumnStats(is_base);
        }
    }
    
    /**
     * @brief Check if a column is registered.
     */
    bool is_registered(gamax::column_view column) const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _column_stats.find(cache_key(column)) != _column_stats.end();
    }
    
    /**
     * @brief Check if a specific statistic has been computed for a column.
     */
    bool has_stat(gamax::column_view column, StatsType type) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _column_stats.find(cache_key(column));
        if (it != _column_stats.end()) {
            return it->second.has_stat(type);
        }
        return false;
    }
    
    /**
     * @brief Get a statistic value for a column.
     */
    std::optional<StatsValue> get_stat(gamax::column_view column, StatsType type) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _column_stats.find(cache_key(column));
        if (it != _column_stats.end()) {
            return it->second.get_stat(type);
        }
        return std::nullopt;
    }
    
    /**
     * @brief Get a scalar statistic value for a column.
     */
    std::optional<gamax::scalar> get_scalar_stat(gamax::column_view column, StatsType type) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _column_stats.find(cache_key(column));
        if (it != _column_stats.end()) {
            return it->second.get_scalar_stat(type);
        }
        return std::nullopt;
    }
    
    /**
     * @brief Get a histogram statistic for a column.
     */
    std::shared_ptr<Histogram> get_histogram_stat(gamax::column_view column, StatsType type) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _column_stats.find(cache_key(column));
        if (it != _column_stats.end()) {
            return it->second.get_histogram_stat(type);
        }
        return nullptr;
    }
    
    /**
     * @brief Store a statistic value for a column.
     * 
     * If the column is not registered, it will be auto-registered as intermediate (is_base=false).
     */
    void set_stat(gamax::column_view column, StatsType type, StatsValue value) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto const key = cache_key(column);
        auto it = _column_stats.find(key);
        if (it == _column_stats.end()) {
            _column_stats[key] = ColumnStats(false);
        }
        _column_stats[key].set_stat(type, std::move(value));
    }
    
    /**
     * @brief Clear statistics for all intermediate columns.
     * 
     * Call this after a query has finished processing to free memory
     * used by intermediate column statistics.
     */
    void clear_intermediate_stats() {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto it = _column_stats.begin(); it != _column_stats.end(); ) {
            if (!it->second.is_base()) {
                it = _column_stats.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    /**
     * @brief Clear statistics for a specific column.
     */
    void clear_column_stats(gamax::column_view column) {
        std::lock_guard<std::mutex> lock(_mutex);
        _column_stats.erase(cache_key(column));
    }
    
    /**
     * @brief Clear all statistics (including base columns).
     */
    void clear_all() {
        std::lock_guard<std::mutex> lock(_mutex);
        _column_stats.clear();
    }
    
    /**
     * @brief Get the number of registered columns.
     */
    size_t num_columns() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _column_stats.size();
    }
    
    /**
     * @brief Get statistics summary for debugging.
     */
    std::string get_stats_summary() const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::string summary = "StatsStorage Summary:\n";
        summary += "  Total columns: " + std::to_string(_column_stats.size()) + "\n";
        
        size_t base_count = 0, intermediate_count = 0;
        for (const auto& [key, stats] : _column_stats) {
            if (stats.is_base()) {
                ++base_count;
            } else {
                ++intermediate_count;
            }
        }
        summary += "  Base columns: " + std::to_string(base_count) + "\n";
        summary += "  Intermediate columns: " + std::to_string(intermediate_count) + "\n";
        return summary;
    }

private:
    StatsStorage() = default;
    ~StatsStorage() = default;
    StatsStorage(const StatsStorage&) = delete;
    StatsStorage& operator=(const StatsStorage&) = delete;
    
    static ColumnCacheKey cache_key(gamax::column_view column) {
        if (column.identity() == 0) {
            throw std::invalid_argument(
                "StatsStorage requires a column_view with a stable logical identity");
        }
        return {column.identity(), column.offset(), column.size(), column.type().id()};
    }

    mutable std::mutex _mutex;
    std::unordered_map<ColumnCacheKey, ColumnStats, ColumnCacheKeyHash> _column_stats;
};

/**
 * @brief RAII helper for managing intermediate column statistics.
 * 
 * Automatically clears intermediate column statistics when destroyed,
 * useful for ensuring cleanup at the end of query processing.
 */
class IntermediateStatsScope {
public:
    IntermediateStatsScope() = default;
    ~IntermediateStatsScope() {
        StatsStorage::instance().clear_intermediate_stats();
    }
    
    // Non-copyable, non-movable
    IntermediateStatsScope(const IntermediateStatsScope&) = delete;
    IntermediateStatsScope& operator=(const IntermediateStatsScope&) = delete;
    IntermediateStatsScope(IntermediateStatsScope&&) = delete;
    IntermediateStatsScope& operator=(IntermediateStatsScope&&) = delete;
};

/**
 * @brief Request to compute specific statistics for a column.
 */
struct StatsRequest {
    gamax::column_view column;
    std::vector<StatsType> stats_types;
    bool is_base{false};  // true if base table column, false if intermediate
    
    // Optional parameters for histogram computation
    gamax::size_type num_buckets{100};  // Default number of buckets for histograms
};

/**
 * @brief Builder for creating StatsRequest objects fluently.
 */
class StatsRequestBuilder {
public:
    explicit StatsRequestBuilder(gamax::column_view column) {
        _request.column = column;
    }
    
    StatsRequestBuilder& as_base() {
        _request.is_base = true;
        return *this;
    }
    
    StatsRequestBuilder& as_intermediate() {
        _request.is_base = false;
        return *this;
    }
    
    StatsRequestBuilder& with_min() {
        _request.stats_types.push_back(StatsType::MIN);
        return *this;
    }
    
    StatsRequestBuilder& with_max() {
        _request.stats_types.push_back(StatsType::MAX);
        return *this;
    }
    
    StatsRequestBuilder& with_avg() {
        _request.stats_types.push_back(StatsType::AVG);
        return *this;
    }
    
    StatsRequestBuilder& with_ndv() {
        _request.stats_types.push_back(StatsType::NDV_APPROX);
        return *this;
    }
    
    StatsRequestBuilder& with_inversion_count() {
        _request.stats_types.push_back(StatsType::INV_COUNT_APPROX);
        return *this;
    }
    
    StatsRequestBuilder& with_equi_width_histogram(gamax::size_type num_buckets = 100) {
        _request.stats_types.push_back(StatsType::EQUI_WIDTH_HISTOGRAM);
        _request.num_buckets = num_buckets;
        return *this;
    }
    
    StatsRequestBuilder& with_equi_depth_histogram(gamax::size_type num_buckets = 100) {
        _request.stats_types.push_back(StatsType::EQUI_DEPTH_HISTOGRAM);
        _request.num_buckets = num_buckets;
        return *this;
    }
    
    StatsRequest build() const {
        return _request;
    }

private:
    StatsRequest _request;
};

/**
 * @brief Convenience function to create a StatsRequestBuilder.
 */
inline StatsRequestBuilder request_stats_for(gamax::column_view column) {
    return StatsRequestBuilder(column);
}

} // namespace gamax::stats
