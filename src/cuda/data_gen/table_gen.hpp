// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <memory>

#include "types/data_type.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"

namespace gamax::datagen::device {
    /**
     * Generate a table with repeated values from a source column.
     * 
     * @param src: Source column to generate the table from.
     * @param card: Cardinality of the table, each source value is repeated card/src.size() times.
     * @param ncols: Number of columns in the table, all columns are the same.
     * @param seed: Seed for the random shuffling of generated columns.
     * @return A unique pointer to the generated table.
     */
    std::unique_ptr<gamax::table>
    generate_table(
        gamax::column_view src,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const int seed,
        cudaStream_t stream);

    /**
     * Generate a table with uniformly distributed primary keys.
     * 
     * @param type: Data type of the primary keys.
     * @param card: Cardinality of the table.
     * @param ncols: Number of columns in the table.
     * @param seed: Seed for the random shuffling of generated columns.
     * @return A unique pointer to the generated table.
     */
    std::unique_ptr<gamax::table>
    pk_uniform_id(
        const gamax::data_type type,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const int seed,
        cudaStream_t stream,
        const uint64_t start = 0);

    /**
     * Generate a table with sparsely distributed primary keys.
     * 
     * @param type: Data type of the primary keys.
     * @param card: Cardinality of the table.
     * @param ncols: Number of columns in the table.
     * @param seed: Seed for the random shuffling of generated columns.
     * @return A unique pointer to the generated table.
     */
    // std::unique_ptr<gamax::table>
    // pk_uniform_sparse(
    //     const gamax::data_type type,
    //     const gamax::size_type card,
    //     const gamax::size_type ncols,
    //     const int seed);
    
    /**
     * Generate a table with foreign keys uniformly distributed from a sequence.
     * 
     * @param seq: Sequence to generate the foreign keys from.
     * @param card: Cardinality of the table.
     * @param ncols: Number of columns in the table.
     * @param match_ratio: Match ratio for the foreign keys.
     * @param seed: Seed for the random shuffling of generated columns.
     * @return A unique pointer to the generated table.
     */
    std::unique_ptr<gamax::table>
    fk_uniform(
        gamax::column_view seq,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const double match_ratio,
        const int seed,
        cudaStream_t stream);
    
    /**
     * Generate a table with uniformly distributed foreign keys.
     * 
     * @param type: Data type of the foreign keys.
     * @param n_unique_keys: Number of unique keys.
     * @param card: Cardinality of the table.
     * @param ncols: Number of columns in the table.
     * @param match_ratio: Match ratio for the foreign keys.
     * @param seed: Seed for the random shuffling of generated columns.
     * @return A unique pointer to the generated table.
     */
    std::unique_ptr<gamax::table>
    fk_uniform_id(
        const gamax::data_type type,
        const gamax::size_type n_unique_keys,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const double match_ratio,
        const int seed,
        cudaStream_t stream,
        const uint64_t start = 0);
    
    /**
     * Generate a table with foreign keys following a Zipfian distribution from a sequence.
     * 
     * @param seq: Sequence to generate the foreign keys from.
     * @param card: Cardinality of the table.
     * @param ncols: Number of columns in the table.
     * @param zipf_factor: Zipfian distribution factor.
     * @param seed: Seed for the random shuffling of generated columns.
     * @return A unique pointer to the generated table.
     */
    std::unique_ptr<gamax::table>
    fk_zipf(
        gamax::column_view seq,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const double zipf_factor,
        const int seed,
        cudaStream_t stream);
    
    /**
     * Generate a table with foreign keys following a Zipfian distribution.
     * 
     * @param type: Data type of the foreign keys.
     * @param n_unique_keys: Number of unique keys.
     * @param card: Cardinality of the table.
     * @param ncols: Number of columns in the table.
     * @param zipf_factor: Zipfian distribution factor.
     * @param seed: Seed for the random shuffling of generated columns.
     * @return A unique pointer to the generated table.
     */
    std::unique_ptr<gamax::table>
    fk_zipf_id(
        const gamax::data_type type,
        const gamax::size_type n_unique_keys,
        const gamax::size_type card,
        const gamax::size_type ncols,
        const double zipf_factor,
        const int seed,
        cudaStream_t stream,
        const uint64_t start = 0);
}
