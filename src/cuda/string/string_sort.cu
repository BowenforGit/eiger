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

namespace gamax {
using namespace gamax::detail;
using prefix_t = int32_t;
constexpr size_t PREFIX_SIZE = sizeof(prefix_t);

namespace {
    template<typename string_col_impl_t>
    struct string_sort_comparator {
        __host__ __device__
        string_sort_comparator(gamax::string_column_view strings, gamax::sort_order order)
            : _strings(strings), is_ascending(order == gamax::sort_order::ASCENDING) {}

        __host__ __device__
       bool operator()(
            const gamax::size_type lhs_idx, 
            const gamax::size_type rhs_idx) const 
        {
            auto lhs = _strings.viewAt(lhs_idx);
            auto rhs = _strings.viewAt(rhs_idx);
            auto min_len = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
            const char* lhs_data = lhs.data();
            const char* rhs_data = rhs.data();

            for(gamax::size_type i = 0; i < min_len; ++i) {
                if(lhs_data[i] < rhs_data[i]) {
                    return final(true);
                } else if(lhs_data[i] > rhs_data[i]) {
                    return final(false);
                }
            }
            
            return final(lhs.size() < rhs.size());
        }
    
    private:
        string_col_impl_t _strings;
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
            gamax::string_column_view strings,
            column_view prefix, gamax::sort_order order)
            : _strings(strings)
            , _prefix(prefix.data<prefix_t>())
            , is_ascending(order == gamax::sort_order::ASCENDING) 
        {
            if(_prefix == nullptr) {
                throw std::runtime_error("Prefix column is null.");
            }

            if(prefix.size() != strings.size()) {
                throw std::runtime_error("Prefix column size does not match string column size.");
            }

            if(prefix.type().id() != gamax::type_to_id<prefix_t>()) {
                throw std::runtime_error("Prefix column type is not prefix_t.");
            }
        }

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

            auto lhs = _strings.viewAt(lhs_idx);
            auto rhs = _strings.viewAt(rhs_idx);
            auto min_len = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
            const char* lhs_data = lhs.data();
            const char* rhs_data = rhs.data();

            for(gamax::size_type i = 0; i < min_len; ++i) {
                if(lhs_data[i] < rhs_data[i]) {
                    return final(true);
                } else if(lhs_data[i] > rhs_data[i]) {
                    return final(false);
                }
            }
            
            return final(lhs.size() < rhs.size());
        }
    
    private:
        string_col_impl_t _strings;
        prefix_t* _prefix;
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
        string_col_impl_t str_impl(in);
        thrust::transform(
            thrust::cuda::par(*get_thrust_allocator()).on(stream),
            itr, itr + in.size(),
            prefix_col->template data<prefix_t>(),
            [str_impl] __device__ (gamax::size_type idx) {
                prefix_t prefix = static_cast<prefix_t>(0);
                auto str = str_impl.viewAt(idx);
                auto len = str.size() < PREFIX_SIZE ? str.size() : PREFIX_SIZE;
                const char* str_data = str.data();
                for(gamax::size_type i = 0; i < len; ++i) {
                    reinterpret_cast<char*>(&prefix)[i] = str_data[i];
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
} // namespace gamax