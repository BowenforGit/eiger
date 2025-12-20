#include "data_gen/string_generator.hpp"

#include <cuda_runtime.h>
#include <functional>
#include <string_view>
#include <memory>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "string/string_column.hpp"
#include "memory/column.hpp"
#include "utils/utils.cuh"
#include "utils/transform.hpp"
#include "memory/mem_manager.hpp"

namespace gamax::datagen::device {
namespace {
    column_ptr generate_string_lengths(
        gamax::size_type num_strings,
        const string_generation_options& options,
        cudaStream_t stream)
    {
        auto lengths = gamax::make_column(
            gamax::data_type{gamax::type_id::INT32},
            num_strings,
            false,
            stream);
        if(options.length_dist == FIXED) {
            thrust::fill_n(
                thrust::cuda::par(*get_thrust_allocator()).on(stream),
                lengths->template data<int32_t>(),
                num_strings,
                static_cast<int32_t>(options.fixed.length));
        }
        else if(options.length_dist == UNIFORM) {
            throw std::runtime_error("UNIFORM length distribution not implemented yet");
        }
        else if(options.length_dist == ZIPFIAN) {
            throw std::runtime_error("ZIPFIAN length distribution not implemented yet");
        }
        return lengths;
    }
    
    column_ptr generate_hit_mask(
        gamax::size_type num_strings,
        const string_generation_options& options,
        cudaStream_t stream)
    {
        auto hit_mask = gamax::make_column(
            gamax::data_type{gamax::type_id::BOOL8},
            num_strings,
            false,
            stream);
        if(options.selectivity == 1.0) {
            thrust::fill_n(
                thrust::cuda::par(*get_thrust_allocator()).on(stream),
                hit_mask->template data<bool>(),
                num_strings,
                true);
        } else {
            throw std::runtime_error("Partial hit rate not implemented yet");
        }
        return hit_mask;
    }
    
    std::pair<column_ptr, int64_t> generate_offsets_from_lengths(
        const column_view& string_lengths,
        cudaStream_t stream)
    {
        auto offsets = util::device::prefix_sum(
            string_lengths, 
            gamax::prefix_sum_options::INCLUSIVE_PAD_ZERO, 
            gamax::data_type{gamax::type_id::INT64},
            stream);
        
        int64_t total_bytes = 0;
        cudaMemcpyAsync(
            &total_bytes, 
            offsets->template data<int64_t>() + offsets->size() - 1, 
            sizeof(int64_t), 
            cudaMemcpyDeviceToHost, 
            stream);

        cudaStreamSynchronize(stream);
        return {std::move(offsets), total_bytes};
    }
    
    struct startswith_fn {
        __device__ startswith_fn(gamax::string_scalar_view pattern)
            : _pattern(pattern) {}

        __device__ void operator()(char* str_ptr, gamax::size_type str_length, bool hit) const {
            if(hit && str_length >= _pattern.size()) {
                #pragma unroll
                for(int32_t i = 0; i < _pattern.size(); ++i) {
                    str_ptr[i] = _pattern[i];
                }
            }
            
            #pragma unroll
            for(int32_t i = (hit ? _pattern.size() : 0); i < str_length; ++i) {
                str_ptr[i] = 'm' + (i % 26);
            }
        }

    private:
        gamax::string_scalar_view _pattern;
    };

    struct endswith_fn {
    __device__ endswith_fn(gamax::string_scalar_view pattern)
        : _pattern(pattern) {}

    __device__ void operator()(char* str_ptr, gamax::size_type str_length, bool hit) const {
        if(hit && str_length >= _pattern.size()) {
            #pragma unroll
            for(int32_t i = str_length - _pattern.size(); i < str_length; ++i) {
                str_ptr[i] = _pattern[i - (str_length - _pattern.size())];
            }
        }
        
        #pragma unroll
        for(int32_t i = 0; i < (hit ? str_length -_pattern.size() : str_length); ++i) {
            str_ptr[i] = 'm' + (i % 26);
        }
    }

    private:
        gamax::string_scalar_view _pattern;
    };

    // so far, we make the contains function just fill the pattern at the end of the string
    // this is to show the worst-case performance of the contains function
    struct contains_fn {
    __device__ contains_fn(gamax::string_scalar_view pattern)
        : _pattern(pattern) {}

    __device__ void operator()(char* str_ptr, gamax::size_type str_length, bool hit) const {
        if(hit && str_length >= _pattern.size()) {
            #pragma unroll
            for(int32_t i = str_length - _pattern.size(); i < str_length; ++i) {
                str_ptr[i] = _pattern[i - (str_length - _pattern.size())];
            }
        }
        
        #pragma unroll
        for(int32_t i = 0; i < (hit ? str_length -_pattern.size() : str_length); ++i) {
            // str_ptr[i] = 'm' + (i % 26);
            str_ptr[i] = _pattern[i % _pattern.size()];
            if(i % _pattern.size() == _pattern.size() - 1) {
                str_ptr[i] += 1; // introduce a mismatch
            }
        }
    }

    private:
        gamax::string_scalar_view _pattern;
    };
    
    template <typename Fn>
    __global__ void generate_strings_kernel(
        char* __restrict__ data,
        const int64_t* __restrict__ offsets,
        gamax::size_type num_strings,
        gamax::string_scalar_view pattern,
        const bool* __restrict__ hit_mask)
    {
        Fn f(pattern);

        for(gamax::size_type idx = get_cuda_tid(); idx < num_strings; idx += nthreads()) {
            auto length = offsets[idx + 1] - offsets[idx];
            char* str_ptr = data + offsets[idx];
            f(str_ptr, length, hit_mask[idx]);
        }   
    }
} // namespace anonymous
    string_column_ptr generate_strings(
        gamax::size_type num_strings,
        std::string_view pattern,
        const string_generation_options& options,
        cudaStream_t stream)
    {
        auto string_lengths = generate_string_lengths(num_strings, options, stream);
        auto hit_mask = generate_hit_mask(num_strings, options, stream);
        auto [offsets, total_bytes] = generate_offsets_from_lengths(string_lengths->view(), stream);

        device_buffer string_data(total_bytes, false, stream);
        auto d_pattern = gamax::make_string_scalar(pattern, stream);
        auto kernel_func = generate_strings_kernel<startswith_fn>;
        if(options.pattern_pos == PREFIX) {
            kernel_func = generate_strings_kernel<startswith_fn>;
        } else if(options.pattern_pos == SUFFIX) {
            kernel_func = generate_strings_kernel<endswith_fn>;
        } else if(options.pattern_pos == SUBSTRING) {
            kernel_func = generate_strings_kernel<contains_fn>;
        } else {
            throw std::runtime_error("Substring pattern position not implemented yet");
        }

        kernel_func<<<num_tb(num_strings, 256), 256, 0, stream>>>( 
            static_cast<char*>(string_data.data()),
            offsets->template data<int64_t>(),
            num_strings,
            d_pattern->view(),
            hit_mask->template data<bool>());
        
        return gamax::make_string_column(
            std::move(string_data),
            std::move(offsets->buffer()),
            num_strings,
            stream,
            options.impl);
    }
} // namespace gamax::datagen::device