#pragma once

#include <cuda_runtime_api.h>
#include <string_view>
#include "string/string_column.hpp"

namespace gamax::datagen::device {
    enum string_length_distribution {
        FIXED,
        UNIFORM,
        ZIPFIAN
    };
    enum pattern_position {
        PREFIX,
        SUFFIX,
        SUBSTRING
    };
    struct string_generation_options {
        string_length_distribution length_dist = FIXED;
        gamax::string_impl_id impl = DEFAULT_STRING_IMPL;
        double selectivity = 1.0; // for prefix/suffix/substring generation
        pattern_position pattern_pos = PREFIX;
        struct fixed_length_options {
            gamax::size_type length = 10;
        };

        struct uniform_length_options {
            gamax::size_type min_length = 5;
            gamax::size_type max_length = 15;
        };

        struct zipfian_length_options {
            gamax::size_type min_length = 5;
            gamax::size_type max_length = 15;
            double zipf_factor = 1.0;
        };

        union {
            fixed_length_options fixed;
            uniform_length_options uniform;
            zipfian_length_options zipfian;
        };
    };

    string_column_ptr generate_strings(
        gamax::size_type num_strings,
        std::string_view pattern,
        const string_generation_options& options,
        cudaStream_t stream = cudaStreamDefault);

    /**
     * @brief Generate a column of random strings for sorting benchmarks.
     * 
     * Unlike generate_strings which creates strings with a common pattern,
     * this function generates strings with random characters, making it
     * suitable for benchmarking sorting operations.
     * 
     * @param num_strings Number of strings to generate
     * @param string_length Length of each string (fixed length)
     * @param impl String implementation to use
     * @param seed Random seed for reproducibility
     * @param stream CUDA stream
     * @return string_column_ptr Column containing random strings
     */
    string_column_ptr generate_random_strings(
        gamax::size_type num_strings,
        gamax::size_type string_length,
        gamax::string_impl_id impl = DEFAULT_STRING_IMPL,
        int seed = 42,
        cudaStream_t stream = cudaStreamDefault);
} // namespace gamax::datagen::device