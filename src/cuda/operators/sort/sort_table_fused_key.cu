#include <memory>
#include <utility>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

#include <cuda.h>
#include <cub/cub.cuh>

#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "operators/sort/sort.hpp"
#include "operators/sort/sort_helpers.cuh"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "memory/table.hpp"
#include "stats/stats_compute.hpp"
#include "compress/dictionary_encode.hpp"

namespace gamax::sort {
namespace {

// =========================================================================
// Configuration constants for key fusion optimization
// =========================================================================

// Minimum bit savings required before considering dictionary encoding.
// If ceil(log2(NDV)) < original_bits - this threshold, use dictionary.
constexpr int kDictEncodeBitSavingsThreshold = 8;

// Maximum total bits we can fuse into a single key (uint64_t)
constexpr int kMaxFusedKeyBits = 64;

// =========================================================================
// Key fusion metadata structure (passed to device via pointer)
// =========================================================================

struct KeyFusionInfo {
    const void* data;           // Pointer to key column data (possibly dictionary-encoded)
    gamax::type_id type;        // Type of the data being read
    int bit_width;              // Number of bits this key occupies in fused key
    int bit_shift;              // Bit position (from LSB) where this key starts
    uint64_t min_value;         // Minimum value (for normalization), stored as uint64
    bool is_descending;         // True if sort order is descending
    bool is_signed;             // True if original type is signed (for sign bit handling)
};

// =========================================================================
// Helper: Compute number of bits needed to represent a range
// =========================================================================

inline int bits_for_range(uint64_t range) {
    if (range == 0) return 0;  // All values are the same
    // ceil(log2(range + 1))
    int bits = 0;
    while ((1ULL << bits) <= range) {
        ++bits;
        if (bits >= 64) break;  // Cap at 64 bits
    }
    return bits;
}

// =========================================================================
// Type dispatcher functor to check if a type is suitable for fusion
// =========================================================================

struct is_fusion_compatible_functor {
    template<typename T>
    bool operator()() const {
        // Only integer types are fusable, no floats or strings
        if constexpr (std::is_same_v<T, gamax::string_type_tag>) {
            return false;
        } else if constexpr (std::is_floating_point_v<T>) {
            return false;
        } else {
            return std::is_integral_v<T>;
        }
    }
};

// =========================================================================
// Type dispatcher functor to compute key statistics and fusion info
// =========================================================================

struct compute_key_fusion_info_functor {
    gamax::column_view column;
    gamax::sort_order order;
    cudaStream_t stream;
    
    // Output parameters
    int* out_bits_needed;
    uint64_t* out_min_value;
    bool* out_is_signed;
    bool* out_use_dict;
    gamax::size_type* out_ndv;  // If dict encoding is recommended
    
    template<typename T, typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
    void operator()() const {
        // Compute min and max
        gamax::scalar min_scalar = gamax::stats::compute_min(column, stream);
        gamax::scalar max_scalar = gamax::stats::compute_max(column, stream);
        
        T min_val = min_scalar.get<T>();
        T max_val = max_scalar.get<T>();
        
        // Calculate the range
        uint64_t range;
        if constexpr (std::is_signed_v<T>) {
            // For signed types, compute range carefully
            // Cast to larger signed type first to avoid overflow
            int64_t min_s = static_cast<int64_t>(min_val);
            int64_t max_s = static_cast<int64_t>(max_val);
            range = static_cast<uint64_t>(max_s - min_s);
            *out_min_value = static_cast<uint64_t>(min_s);  // Store as uint64 (bit pattern)
            *out_is_signed = true;
        } else {
            range = static_cast<uint64_t>(max_val) - static_cast<uint64_t>(min_val);
            *out_min_value = static_cast<uint64_t>(min_val);
            *out_is_signed = false;
        }
        
        int original_bits = sizeof(T) * 8;
        int range_bits = bits_for_range(range);
        
        // Check if dictionary encoding would help
        int bit_savings = original_bits - range_bits;
        
        if (bit_savings < kDictEncodeBitSavingsThreshold) {
            // Range compression isn't great, try NDV
            gamax::scalar ndv_scalar = gamax::stats::compute_ndv_approx(column, stream);
            gamax::size_type ndv = ndv_scalar.get<gamax::size_type>();
            
            int ndv_bits = bits_for_range(static_cast<uint64_t>(ndv));
            
            if (original_bits - ndv_bits >= kDictEncodeBitSavingsThreshold) {
                // Dictionary encoding is beneficial
                *out_bits_needed = ndv_bits;
                *out_use_dict = true;
                *out_ndv = ndv;
                return;
            }
        }
        
        // Use range compression
        *out_bits_needed = range_bits;
        *out_use_dict = false;
        *out_ndv = 0;
    }

    template<typename T, typename std::enable_if_t<!std::is_integral_v<T>, int> = 0>
    void operator()() const {
        throw std::runtime_error("Non-integral types are not supported for key fusion");
    }
};

// =========================================================================
// Device functor to read and normalize a value for key fusion
// =========================================================================

struct read_and_normalize_functor {
    template<typename T>
    __device__ __forceinline__
    uint64_t operator()(const void* data, gamax::size_type idx, uint64_t min_value) const {
        if constexpr (std::is_signed_v<T>) {
            T v = static_cast<const T*>(data)[idx];
            int64_t normalized = static_cast<int64_t>(v) - static_cast<int64_t>(min_value);
            return static_cast<uint64_t>(normalized);
        } else {
            T v = static_cast<const T*>(data)[idx];
            return static_cast<uint64_t>(v) - min_value;
        }
    }
};

// =========================================================================
// Key fusion kernel: Fuse multiple keys into a single uint64_t key
// =========================================================================

__global__ void fuse_keys_kernel(
    const KeyFusionInfo* __restrict__ key_infos,
    int num_keys,
    gamax::size_type num_rows,
    uint64_t* __restrict__ fused_keys)
{
    gamax::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_rows) return;
    
    uint64_t fused = 0;
    
    for (int k = 0; k < num_keys; ++k) {
        const KeyFusionInfo& info = key_infos[k];
        
        if (info.bit_width == 0) continue;  // Key has single value, skip
        
        // Read and normalize value using type dispatcher
        uint64_t value = gamax::type_dispatcher(
            info.type, 
            read_and_normalize_functor{}, 
            info.data, idx, info.min_value);
        
        // Mask to bit_width
        uint64_t mask = (info.bit_width >= 64) ? ~0ULL : ((1ULL << info.bit_width) - 1);
        value &= mask;
        
        // For descending order, flip all bits within the bit_width
        if (info.is_descending) {
            value = mask - value;
        }
        
        // Shift into position and OR into fused key
        fused |= (value << info.bit_shift);
    }
    
    fused_keys[idx] = fused;
}

// =========================================================================
// Main fused key sort implementation
// =========================================================================

std::unique_ptr<gamax::table>
sort_table_fused_keys(
    gamax::table_view const& table,
    std::vector<int> const& sort_keys,
    std::vector<gamax::sort_order> const& sort_orders,
    cudaStream_t stream)
{
    const auto num_rows = table.num_rows();
    const int num_keys = static_cast<int>(sort_keys.size());
    
    if (num_rows == 0) {
        return detail::make_empty_table_like(table, stream);
    }
    
    // Step 1: Check if all keys are fusion-compatible
    for (int k = 0; k < num_keys; ++k) {
        auto col = table.get_column(sort_keys[k]);
        bool compatible = gamax::type_dispatcher(col.type(), is_fusion_compatible_functor{});
        if (!compatible) {
            // Fall back to default multi-key sort
            return sort_table(table, sort_keys, sort_orders, stream);
        }
    }
    
    // Step 2: Compute fusion info for each key
    std::vector<KeyFusionInfo> host_key_infos(num_keys);
    std::vector<std::unique_ptr<gamax::column>> dict_encoded_columns;  // Keep alive
    
    int total_bits = 0;
    
    for (int k = 0; k < num_keys; ++k) {
        auto col = table.get_column(sort_keys[k]);
        
        int bits_needed = 0;
        uint64_t min_value = 0;
        bool is_signed = false;
        bool use_dict = false;
        gamax::size_type ndv = 0;
        
        gamax::type_dispatcher(col.type(), compute_key_fusion_info_functor{
            col, sort_orders[k], stream,
            &bits_needed, &min_value, &is_signed, &use_dict, &ndv
        });
        
        const void* data_ptr = col.data<void>();
        gamax::type_id data_type = col.type().id();
        
        // Apply dictionary encoding if beneficial
        if (use_dict) {
            // Need to get the actual column for dictionary encoding
            // Dictionary encode expects column reference, so we create a temporary
            auto temp_col = gamax::make_column(col.type(), col.size(), false, stream);
            cudaMemcpyAsync(temp_col->data<void>(), col.data<void>(), 
                          col.size() * gamax::size_of(col.type()),
                          cudaMemcpyDeviceToDevice, stream);
            
            auto [encoded_col, dict_size] = gamax::compress::dictionary_encode(
                *temp_col, gamax::data_type{gamax::type_id::INT32}, ndv, 0.25, stream);
            
            // Update fusion info to use encoded column
            data_ptr = encoded_col->data<void>();
            data_type = encoded_col->type().id();
            min_value = 0;  // Dictionary indices start at 0
            is_signed = false;
            bits_needed = bits_for_range(static_cast<uint64_t>(dict_size));
            
            dict_encoded_columns.push_back(std::move(encoded_col));
        }
        
        // Check if we exceed max bits
        if (total_bits + bits_needed > kMaxFusedKeyBits) {
            // Cannot fuse all keys, fall back
            return sort_table(table, sort_keys, sort_orders, stream);
        }
        
        // Fill in KeyFusionInfo
        // Keys are packed from MSB to LSB for correct comparison order
        // (most significant key at highest bits)
        host_key_infos[k].data = data_ptr;
        host_key_infos[k].type = data_type;
        host_key_infos[k].bit_width = bits_needed;
        host_key_infos[k].min_value = min_value;
        host_key_infos[k].is_descending = (sort_orders[k] == gamax::sort_order::DESCENDING);
        host_key_infos[k].is_signed = is_signed;
        // bit_shift will be set after we know all bit widths
        
        total_bits += bits_needed;
    }
    
    // Calculate bit shifts: first key at highest position
    int current_shift = total_bits;
    for (int k = 0; k < num_keys; ++k) {
        current_shift -= host_key_infos[k].bit_width;
        host_key_infos[k].bit_shift = current_shift;
    }
    
    // Step 3: Copy KeyFusionInfo array to device
    KeyFusionInfo* d_key_infos = nullptr;
    allocate_mem(reinterpret_cast<void**>(&d_key_infos), false, 
                num_keys * sizeof(KeyFusionInfo), stream);
    cudaMemcpyAsync(d_key_infos, host_key_infos.data(), 
                   num_keys * sizeof(KeyFusionInfo),
                   cudaMemcpyHostToDevice, stream);
    
    // Step 4: Allocate fused keys column
    auto fused_keys = gamax::make_column<uint64_t>(num_rows, stream);
    
    // Step 5: Launch fusion kernel
    constexpr int kBlockSize = 256;
    int num_blocks = (num_rows + kBlockSize - 1) / kBlockSize;
    
    fuse_keys_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        d_key_infos,
        num_keys,
        num_rows,
        fused_keys->data<uint64_t>()
    );
    
    release_mem(d_key_infos);
    
    // Step 6: Use argsort to sort fused keys and get sorted indices
    auto [sorted_fused_keys, sorted_indices] = argsort(
        fused_keys->view(), 0, total_bits, 
        gamax::sort_order::ASCENDING, nullptr, 0, stream);
    
    // Step 7: Apply sorted indices to reorder all columns using gather
    return detail::create_sorted_table_from_indices(table, sorted_indices->view(), stream);
}

} // anonymous namespace

std::unique_ptr<gamax::table>
sort_table(
    gamax::table_view const& table,
    std::vector<int> const& sort_keys,
    std::vector<gamax::sort_order> const& sort_orders,
    sort_table_options const& options,
    cudaStream_t stream)
{
    if (!options.smart_keys_fusion) {
        // Fallback to default implementation
        return sort_table(table, sort_keys, sort_orders, stream);
    }
    
    // Try fused key optimization
    return sort_table_fused_keys(table, sort_keys, sort_orders, stream);
}

} // namespace gamax::sort