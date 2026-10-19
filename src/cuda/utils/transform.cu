// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "transform.hpp"

#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h> 
#include <cub/device/device_select.cuh>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "utils/sequence.cuh"
#include "cub_wrapper.cuh"

namespace gamax {
namespace util {
namespace device {
namespace {
    struct gather_impl {
        template <typename T, typename GatherMapT,
                  typename std::enable_if<std::is_integral<GatherMapT>::value, int>::type = 0>
        void operator()(
            gamax::column_view const& gather_map, 
            gamax::column_view const& input, 
            gamax::mutable_column_view output, 
            cudaStream_t stream)
        {
            const auto N = gather_map.size();

            thrust::device_ptr<const GatherMapT> gather_map_ptr(gather_map.data<GatherMapT>());
            thrust::device_ptr<const T> input_ptr(input.data<T>());
            thrust::device_ptr<T> output_ptr(output.data<T>());

            auto policy = gamax::rmm_exec_policy_handle{stream};
            thrust::gather(
                policy.get(), gather_map_ptr, gather_map_ptr + N, input_ptr, output_ptr);
        }

        template <typename T, typename GatherMapT,
                  typename std::enable_if<!std::is_integral<GatherMapT>::value, int>::type = 0>
        void operator()(
            gamax::column_view const& gather_map, 
            gamax::column_view const& input, 
            gamax::mutable_column_view output, 
            cudaStream_t stream)
        {
            throw std::runtime_error("gather_impl: gather_map type must be an integral type");
        }
    };

    struct shuffle_impl {
        template <typename T>
        void operator()(
            gamax::column_view const& input, 
            gamax::mutable_column_view output, 
            const int seed,
            cudaStream_t stream)
        {
            thrust::default_random_engine g(seed);
            thrust::shuffle_copy(
                gamax::rmm_exec_policy_handle{stream}.get(),
                input.data<T>(), input.data<T>() + input.size(), 
                output.data<T>(), g);
        }
    };

    struct compact_impl {
        template <typename T>
        auto operator()(
            gamax::column_view const& input, 
            gamax::column_view const& mask, 
            gamax::size_type output_size,
            cudaStream_t stream)
        {
            auto output = gamax::make_column<T>(output_size, stream);
            auto d_output_size = gamax::make_column<gamax::size_type>(1, stream);

            CALL_CUB_FUNC(stream, cub::DeviceSelect::Flagged,
                input.template data<T>(),
                mask.template data<bool>(),
                output->template data<T>(),
                d_output_size->template data<gamax::size_type>(),
                input.size(),
                stream);

            return output;
        }
    };

    struct prefix_sum_impl {
        template <typename T, typename U>
        void operator()(
        gamax::column_view const& input,
        gamax::mutable_column_view output,
        gamax::prefix_sum_options option,
        cudaStream_t stream)
        {
            auto policy_owner = gamax::rmm_exec_policy_handle{stream};
            auto& policy = policy_owner.get();
            auto in_ptr = input.data<T>();
            auto out_ptr = output.data<U>();
            auto n = input.size();

            auto itr_in = thrust::make_transform_iterator(
                in_ptr,
                [] __host__ __device__ (T val) { return static_cast<U>(val); });

            // Empty-input short-circuit. thrust/CUB's scan determines
            // temp-storage sizes via a CUDA call, which on some
            // cuda/thrust version combinations reports
            // `cudaErrorInvalidDevice` when the range is 0 elements
            // — the bug that surfaced downstream of an empty join
            // result in Q10 before the builder was fixed. Handling
            // n==0 here covers every caller (gather_strings,
            // string_factory, sort offsets) without them having to
            // guard individually. INCLUSIVE_PAD_ZERO still owes a
            // single zero in out[0]; the other options produce a
            // 0-length output and there's nothing to write.
            if (n == 0) {
                if (option == INCLUSIVE_PAD_ZERO) {
                    cudaMemsetAsync(out_ptr, 0, sizeof(U), stream);
                }
                return;
            }

            switch(option) {
                case INCLUSIVE:
                thrust::inclusive_scan(gamax::rmm_exec_policy_handle{stream}.get(), itr_in, itr_in + n, out_ptr);
                break;
                case EXCLUSIVE:
                thrust::exclusive_scan(gamax::rmm_exec_policy_handle{stream}.get(), itr_in, itr_in + n, out_ptr);
                break;
                case INCLUSIVE_PAD_ZERO:
                cudaMemsetAsync(out_ptr, 0, sizeof(U), stream);
                thrust::inclusive_scan(gamax::rmm_exec_policy_handle{stream}.get(), itr_in, itr_in + n, out_ptr + 1);
                break;
            }
        }
    };

    struct count_nonzero_impl {
        template <typename T>
        gamax::size_type operator()(
            gamax::column_view const& input,
            cudaStream_t stream)
        {
            auto itr = thrust::make_transform_iterator(
                input.data<T>(),
                [] __host__ __device__ (T val) -> gamax::size_type 
                    { return val != static_cast<T>(0) ? static_cast<gamax::size_type>(1) : static_cast<gamax::size_type>(0); });
            
            return thrust::reduce(
                gamax::rmm_sync_exec_policy_handle{stream}.get(),
                itr, 
                itr + input.size(), 
                static_cast<gamax::size_type>(0),
                thrust::plus<gamax::size_type>());
        }
    };
}

    void gather_into(
        gamax::column_view gather_map,
        gamax::column_view input,
        gamax::mutable_column_view output,
        cudaStream_t stream)
    {
        if (output.type() != input.type()) {
            throw std::invalid_argument("gather_into output type differs from input type");
        }
        if (output.size() < gather_map.size()) {
            throw std::invalid_argument("gather_into output is smaller than gather map");
        }
        double_type_dispatcher(
            input.type(), gather_map.type(), gather_impl{}, gather_map, input, output, stream);
    }

    std::unique_ptr<gamax::column> gather(
        gamax::column_view gather_map,
        gamax::column_view input,
        cudaStream_t stream)
    {
        auto output = std::make_unique<gamax::column>(input.type(), gather_map.size(), stream);
        gather_into(gather_map, input, output->mutable_view(), stream);
        return output;
    }

    std::unique_ptr<gamax::column> shuffle(
        gamax::column_view input,
        cudaStream_t stream,
        const int seed)
    {
        auto output = std::make_unique<gamax::column>(input.type(), input.size(), stream);
        type_dispatcher(input.type(), shuffle_impl{}, input, output->mutable_view(), seed, stream);
        return output;
    }

    std::unique_ptr<gamax::column> compact( 
        gamax::column_view input, 
        gamax::column_view mask, 
        cudaStream_t stream)
    {
        auto output_size = thrust::reduce(
            gamax::rmm_sync_exec_policy_handle{stream}.get(),
            mask.data<bool>(), 
            mask.data<bool>() + mask.size(), 
            0);
        
        return compact(input, mask, output_size, stream);
    }

    std::unique_ptr<gamax::column> compact( 
        gamax::column_view input, 
        gamax::column_view mask,
        gamax::size_type output_size,
        cudaStream_t stream)
    {
        return type_dispatcher(input.type(), compact_impl{}, input, mask, output_size, stream);
    }

    std::unique_ptr<gamax::table> compact(
        gamax::table_view input,
        gamax::column_view mask,
        cudaStream_t stream)
    {
        if(input.num_columns() == 0) {
            throw std::runtime_error("compact: input table must have at least one column");
        }

        // STRING columns can't go through the numeric column-`compact`
        // (its `type_dispatcher` excludes STRING). If the input contains
        // any STRING column, mirror `phys::apply_bitmask`: compact an
        // identity tid sequence once, then route numeric columns through
        // gather + STRING columns through `gather_strings`. For pure
        // numeric input we keep the original faster two-phase path that
        // avoids recomputing the mask's popcount.
        bool any_string = false;
        for (gamax::size_type i = 0; i < input.num_columns(); ++i) {
            if (input[i].type().id() == gamax::type_id::STRING) {
                any_string = true;
                break;
            }
        }

        auto res = std::make_unique<gamax::table>();

        if (!any_string) {
            for(gamax::size_type i = 0; i < input.num_columns(); i++) {
                if(i == 0) {
                    res->add_column(compact(input[i], mask, stream));
                } else {
                    res->add_column(compact(input[i], mask, res->num_rows(), stream));
                }
            }
            return res;
        }

        // Mixed path: materialise surviving tids once, then gather each column.
        auto seq = gamax::make_sequence<gamax::size_type>(
            input.num_rows(), stream, 0, 1);
        auto compact_tids = compact(seq->view(), mask, stream);
        auto map_view = compact_tids->view();

        for(gamax::size_type i = 0; i < input.num_columns(); i++) {
            auto col = input[i];
            if (col.type().id() == gamax::type_id::STRING) {
                gamax::string_column_view sv(col, gamax::ARROW_STRING);
                auto gathered = gamax::gather_strings(sv, map_view, stream);
                res->add_column(std::unique_ptr<gamax::column>(std::move(gathered)));
            } else {
                res->add_column(gather(map_view, col, stream));
            }
        }
        return res;
    }

    std::unique_ptr<gamax::column> prefix_sum( 
        gamax::column_view input, 
        gamax::prefix_sum_options option,
        gamax::data_type output_type,
        cudaStream_t stream)
    {
        auto output_size = input.size() + ((option == INCLUSIVE_PAD_ZERO) ? 1 : 0);
        auto output = gamax::make_column(output_type, output_size, stream);

        double_type_dispatcher(input.type(), output_type, prefix_sum_impl{}, input, output->mutable_view(), option, stream);
        return output;
    }

    gamax::size_type count_nonzero(
        gamax::column_view input,
        cudaStream_t stream)
    {
        return type_dispatcher(input.type(), count_nonzero_impl{}, input, stream);
    }

    namespace {
    __global__ void bitmask_and_kernel(const bool* __restrict__ lhs, const bool* __restrict__ rhs, bool* __restrict__ out, gamax::size_type n) {
        gamax::size_type idx = static_cast<gamax::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = lhs[idx] && rhs[idx];
        }
    }

    __global__ void bitmask_or_kernel(const bool* __restrict__ lhs, const bool* __restrict__ rhs, bool* __restrict__ out, gamax::size_type n) {
        gamax::size_type idx = static_cast<gamax::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = lhs[idx] || rhs[idx];
        }
    }

    __global__ void bitmask_not_kernel(const bool* __restrict__ in, bool* __restrict__ out, gamax::size_type n) {
        gamax::size_type idx = static_cast<gamax::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = !in[idx];
        }
    }
    } // namespace

    std::unique_ptr<gamax::column> bitmask_and(
        gamax::column_view lhs,
        gamax::column_view rhs,
        cudaStream_t stream)
    {
        auto n = lhs.size();
        auto output = gamax::make_column(
            gamax::data_type{gamax::type_id::BOOL8}, n, stream);
        constexpr int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        bitmask_and_kernel<<<grid_size, block_size, 0, stream>>>(
            lhs.data<bool>(), rhs.data<bool>(), output->data<bool>(), n);
        return output;
    }

    std::unique_ptr<gamax::column> bitmask_or(
        gamax::column_view lhs,
        gamax::column_view rhs,
        cudaStream_t stream)
    {
        auto n = lhs.size();
        auto output = gamax::make_column(
            gamax::data_type{gamax::type_id::BOOL8}, n, stream);
        constexpr int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        bitmask_or_kernel<<<grid_size, block_size, 0, stream>>>(
            lhs.data<bool>(), rhs.data<bool>(), output->data<bool>(), n);
        return output;
    }

    std::unique_ptr<gamax::column> bitmask_not(
        gamax::column_view input,
        cudaStream_t stream)
    {
        auto n = input.size();
        auto output = gamax::make_column(
            gamax::data_type{gamax::type_id::BOOL8}, n, stream);
        constexpr int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        bitmask_not_kernel<<<grid_size, block_size, 0, stream>>>(
            input.data<bool>(), output->data<bool>(), n);
        return output;
    }

} // namespace device
} // namespace util
} // namespace gamax
