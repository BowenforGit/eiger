#include <memory>
#include <vector>

#include "types/data_type.hpp"
#include "operators/sort/sort.hpp"
#include "operators/sort/sort_helpers.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "compress/multikey_compress.hpp"

namespace gamax::sort {
namespace {

// =========================================================================
// Fused key sort: compress multiple sort keys into one column, then argsort
// =========================================================================

std::unique_ptr<gamax::table>
sort_table_fused_keys(
    gamax::table_view const& table,
    std::vector<int> const& sort_keys,
    std::vector<gamax::sort_order> const& sort_orders,
    cudaStream_t stream)
{
    const auto num_rows = table.num_rows();
    
    if (num_rows == 0) {
        return detail::make_empty_table_like(table, stream);
    }
    
    // Build a key-only table_view from the selected sort key columns
    std::vector<gamax::size_type> key_indices(sort_keys.begin(), sort_keys.end());
    auto key_table = table.select_columns(key_indices);
    
    // Use multikey_compress to fuse all sort keys into a single uint64_t column
    int total_bits = 0;
    gamax::compress::multikey_compress_options compress_opts;
    compress_opts.always_compress = true;
    compress_opts.out_total_bits = &total_bits;
    
    auto fused_keys = gamax::compress::multikey_compress(
        key_table, sort_orders, compress_opts, stream);
    
    if (!fused_keys) {
        // Cannot fuse all keys, fall back to default multi-key sort
        return sort_table(table, sort_keys, sort_orders, stream);
    }
    
    // Argsort the fused keys (always ascending since descending is
    // already handled by bit-flipping inside multikey_compress)
    auto [sorted_fused_keys, sorted_indices] = argsort(
        fused_keys->view(), 0, total_bits, 
        gamax::sort_order::ASCENDING, nullptr, 0, stream);
    
    // Apply sorted indices to reorder all columns using gather
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