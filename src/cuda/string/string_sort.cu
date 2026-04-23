#include <cuda.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cooperative_groups.h>
#include <cub/device/device_merge_sort.cuh>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include "string/string.hpp"
#include "string/string_column.hpp"
#include "string/string_impl.cuh"
#include "types/vec_type.hpp"
#include "utils/utils.cuh"
#include "utils/cub_wrapper.cuh"
#include "memory/mem_manager.hpp"
#include "utils/transform.hpp"

namespace gamax {
using namespace gamax::detail;
using prefix_t = int32_t; // TODO: this better to be unsigned
constexpr size_t PREFIX_SIZE = sizeof(prefix_t);

namespace {
    template<typename string_col_impl_t, typename buffer_index_type>
    __global__ void gather_strings_kernel(
        gamax::string_column_view in,
        const gamax::size_type* gather_map_data,
        char* out_data_ptr,
        const buffer_index_type* out_offsets_ptr,
        gamax::size_type num_strings)
    {
        gamax::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_strings) return;

        const auto gather_idx = gather_map_data[idx];
        string_col_impl_t str_impl(in);
        auto str = str_impl.viewAt(gather_idx);
        const char* str_data = str.data();
        const auto str_len = str.size();

        const auto out_offset = out_offsets_ptr[idx];
        char* out_str_ptr = out_data_ptr + out_offset;
        for(int i = 0; i < str_len; ++i) {
            out_str_ptr[i] = str_data[i];
        }
    }


    template<typename string_col_impl_t>
    struct string_sort_comparator {
        __host__ __device__
        string_sort_comparator(gamax::string_column_view const& strings, gamax::sort_order order)
            : _strings(strings), is_ascending(order == gamax::sort_order::ASCENDING) {}

        __host__ __device__
        bool operator()(
            const gamax::size_type lhs_idx, 
            const gamax::size_type rhs_idx) const 
        {
            string_col_impl_t impl(_strings);
            auto lhs = impl.viewAt(lhs_idx);
            auto rhs = impl.viewAt(rhs_idx);
            auto min_len = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
            const char* lhs_data = lhs.data();
            const char* rhs_data = rhs.data();

            for(int i = 0; i < min_len; ++i) {
                if(lhs_data[i] < rhs_data[i]) {
                    return final(true);
                } else if(lhs_data[i] > rhs_data[i]) {
                    return final(false);
                }
            }
            
            return final(lhs.size() < rhs.size());
        }
    
    private:
        const string_column_view _strings;
        bool is_ascending;

        __host__ __device__
        bool final(bool cmp_result) const {
            return is_ascending ? cmp_result : !cmp_result;
        }
    };

    template<typename string_col_impl_t>
    struct string_sort_comparator_fast {
        __host__ __device__
        string_sort_comparator_fast(
            gamax::string_column_view const& strings,
            column_view prefix, 
            gamax::sort_order order)
            : _strings(strings)
            , _prefix(prefix.data<prefix_t>())
            , is_ascending(order == gamax::sort_order::ASCENDING) {}

        __host__ __device__
        bool operator()(
            const gamax::size_type lhs_idx, 
            const gamax::size_type rhs_idx) const 
        {
            const auto lhs_prefix = _prefix[lhs_idx];
            const auto rhs_prefix = _prefix[rhs_idx];
            if(lhs_prefix < rhs_prefix) {
                return final(true);
            } else if(lhs_prefix > rhs_prefix) {
                return final(false);
            }

            string_col_impl_t impl(_strings);
            auto lhs = impl.viewAt(lhs_idx);
            auto rhs = impl.viewAt(rhs_idx);
            auto min_len = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
            const char* lhs_data = lhs.data();
            const char* rhs_data = rhs.data();

            for(int i = PREFIX_SIZE; i < min_len; ++i) {
                if(lhs_data[i] < rhs_data[i]) {
                    return final(true);
                } else if(lhs_data[i] > rhs_data[i]) {
                    return final(false);
                }
            }
            
            return final(lhs.size() < rhs.size());
        }
    
    private:
        const string_column_view _strings;
        const prefix_t* _prefix;
        bool is_ascending;

        __host__ __device__
        bool final(bool cmp_result) const {
            return is_ascending ? cmp_result : !cmp_result;
        }
    };
} // namespace anonymous

    column_ptr argsort_strings(
        gamax::string_column_view in,
        gamax::sort_order order,
        cudaStream_t stream,
        string_op_options options)
    {
        using string_col_impl_t = typename id_to_string_impl<ARROW_STRING>::string_col_impl;

        auto tid = gamax::make_sequence<gamax::size_type>(in.size(), 0, 1, stream);

        if(!options.use_prefix_sort) {
            string_sort_comparator<string_col_impl_t> comparator(in, order);
            CALL_CUB_FUNC(
                cub::DeviceMergeSort::SortKeys, 
                tid->template data<gamax::size_type>(),
                in.size(),
                comparator,
                stream);
            
            cudaStreamSynchronize(stream);
            return tid;
        }

        auto prefix_col = gamax::make_column<prefix_t>(in.size(), stream);;
        auto itr = thrust::make_counting_iterator<gamax::size_type>(0);
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            itr, itr + in.size(),
            prefix_col->template data<prefix_t>(),
            [in] __device__ (gamax::size_type idx) {
                prefix_t prefix = static_cast<prefix_t>(0);
                string_col_impl_t str_impl(in);
                auto str = str_impl.viewAt(idx);
                auto len = str.size() < PREFIX_SIZE ? str.size() : PREFIX_SIZE;
                const char* str_data = str.data();
                for(gamax::size_type i = 0; i < len; ++i) {
                    reinterpret_cast<char*>(&prefix)[PREFIX_SIZE - 1 - i] = str_data[i];
                }
                return prefix;
            });
        
        string_sort_comparator_fast<string_col_impl_t> comparator(in, prefix_col->view(), order);
        CALL_CUB_FUNC(
            cub::DeviceMergeSort::SortKeys, 
            tid->template data<gamax::size_type>(),
            in.size(),
            comparator,
            stream);

        cudaStreamSynchronize(stream);
        return tid;
    }

    string_column_ptr sort_strings(
        gamax::string_column_view in,
        gamax::sort_order order,
        cudaStream_t stream,
        string_op_options options)
    {
        auto sorted_indices = argsort_strings(in, order, stream, options);
        return gather_strings(in, sorted_indices->view(), stream, options);
    }

    string_column_ptr gather_strings(
        gamax::string_column_view in,
        gamax::column_view const& gather_map,
        cudaStream_t stream,
        string_op_options options)
    {
        using string_col_impl_t = typename id_to_string_impl<ARROW_STRING>::string_col_impl;
        using len_type = typename string_col_impl_t::LengthType;
        using buffer_index_type = typename string_col_impl_t::header_entry_type::buffer_offset_type;

        const buffer_index_type* offsets = reinterpret_cast<const buffer_index_type*>(in.meta());

        auto out_strlen = gamax::make_column<len_type>(gather_map.size(), stream);
        auto itr = thrust::make_counting_iterator<gamax::size_type>(0);

        const auto gather_map_data = gather_map.data<gamax::size_type>();
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            itr, itr + gather_map.size(),
            out_strlen->template data<len_type>(),
            [gather_map_data, offsets] __device__ (gamax::size_type idx) {
                const auto gather_idx = gather_map_data[idx];
                const auto str_len = offsets[gather_idx + 1] - offsets[gather_idx];
                return static_cast<len_type>(str_len);
            });
        
        auto out_offsets = gamax::util::device::prefix_sum(
            out_strlen->view(), 
            gamax::prefix_sum_options::INCLUSIVE_PAD_ZERO,
            gamax::data_type{type_to_id<buffer_index_type>()},
            stream);
        gamax::size_type total_bytes = 0;
        cudaMemcpyAsync(
            &total_bytes, 
            out_offsets->data<buffer_index_type>() + gather_map.size(),
            sizeof(buffer_index_type),
            cudaMemcpyDeviceToHost,
            stream);
        cudaStreamSynchronize(stream);
        
        gamax::device_buffer out_data(total_bytes, false, stream);
        char* out_data_ptr = reinterpret_cast<char*>(out_data.data());
        const buffer_index_type* out_offsets_ptr = out_offsets->data<buffer_index_type>();
        
        constexpr int block_size = 256;
        int num_blocks = (gather_map.size() + block_size - 1) / block_size;
        gather_strings_kernel<string_col_impl_t, buffer_index_type><<<num_blocks, block_size, 0, stream>>>(
            in, gather_map_data, out_data_ptr, out_offsets_ptr, gather_map.size());
        
        cudaStreamSynchronize(stream);
        return gamax::make_string_column(
            std::move(out_data),
            std::move(out_offsets->buffer()),
            gather_map.size(),
            stream,
            ARROW_STRING);
    }

    namespace {
        __global__ void substring_kernel(
            const char* in_data,
            const gamax::size_type* in_offsets,
            char* out_data,
            const gamax::size_type* out_offsets,
            gamax::size_type start,
            gamax::size_type n)
        {
            gamax::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            const auto out_start = out_offsets[idx];
            const auto out_end   = out_offsets[idx + 1];
            const auto out_len   = out_end - out_start;
            if (out_len <= 0) return;

            const auto in_start = in_offsets[idx] + start;
            char* out_ptr = out_data + out_start;
            const char* in_ptr = in_data + in_start;
            for (gamax::size_type i = 0; i < out_len; ++i) {
                out_ptr[i] = in_ptr[i];
            }
        }
    } // namespace

    string_column_ptr substring(
        gamax::string_column_view in,
        gamax::size_type sql_offset,
        gamax::size_type length,
        cudaStream_t stream)
    {
        // SQL offsets ≤ 0 are clamped to 1 (SQL-92 behavior is
        // implementation-defined; TPC-H only uses 1-based positive
        // offsets, so we don't bother emulating negative-offset
        // semantics). Negative length is an error.
        if (length < 0) {
            throw std::runtime_error("substring: negative length");
        }
        const gamax::size_type start = sql_offset > 0 ? (sql_offset - 1) : 0;
        const auto n = in.size();

        using string_col_impl_t = typename id_to_string_impl<ARROW_STRING>::string_col_impl;
        using buffer_index_type = typename string_col_impl_t::header_entry_type::buffer_offset_type;

        const buffer_index_type* in_offsets =
            reinterpret_cast<const buffer_index_type*>(in.meta());

        // Output string lengths: clamp against remaining row width and
        // the caller-provided `length`.
        auto out_strlen = gamax::make_column<buffer_index_type>(n, stream);
        auto itr = thrust::make_counting_iterator<gamax::size_type>(0);
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            itr, itr + n,
            out_strlen->template data<buffer_index_type>(),
            [in_offsets, start, length] __device__ (gamax::size_type idx) {
                const auto row_len = in_offsets[idx + 1] - in_offsets[idx];
                if (start >= row_len) return static_cast<buffer_index_type>(0);
                const auto avail = row_len - start;
                return static_cast<buffer_index_type>(
                    avail < length ? avail : length);
            });

        auto out_offsets = gamax::util::device::prefix_sum(
            out_strlen->view(),
            gamax::prefix_sum_options::INCLUSIVE_PAD_ZERO,
            gamax::data_type{type_to_id<buffer_index_type>()},
            stream);

        gamax::size_type total_bytes = 0;
        cudaMemcpyAsync(
            &total_bytes,
            out_offsets->data<buffer_index_type>() + n,
            sizeof(buffer_index_type),
            cudaMemcpyDeviceToHost,
            stream);
        cudaStreamSynchronize(stream);

        gamax::device_buffer out_data(total_bytes, false, stream);
        char* out_data_ptr = reinterpret_cast<char*>(out_data.data());
        const buffer_index_type* out_offsets_ptr =
            out_offsets->data<buffer_index_type>();
        const char* in_data_ptr = in.template data<char>();

        if (n > 0) {
            constexpr int block_size = 256;
            const int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
            substring_kernel<<<num_blocks, block_size, 0, stream>>>(
                in_data_ptr, in_offsets,
                out_data_ptr, out_offsets_ptr,
                start, n);
        }

        cudaStreamSynchronize(stream);
        return gamax::make_string_column(
            std::move(out_data),
            std::move(out_offsets->buffer()),
            n,
            stream,
            ARROW_STRING);
    }

} // namespace gamax