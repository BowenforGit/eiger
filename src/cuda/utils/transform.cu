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
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
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

            thrust::gather(thrust::device.on(stream), gather_map_ptr, gather_map_ptr + N, input_ptr, output_ptr);
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
                thrust::cuda::par(*get_thrust_allocator()).on(stream), 
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
            auto output = gamax::make_column<T>(output_size);
            auto d_output_size = gamax::make_column<gamax::size_type>(1);

            CALL_CUB_FUNC(cub::DeviceSelect::Flagged, 
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
            const auto& policy = thrust::cuda::par(*get_thrust_allocator()).on(stream);
            auto in_ptr = input.data<T>();
            auto out_ptr = output.data<U>();
            auto n = input.size();

            auto itr_in = thrust::make_transform_iterator(
                in_ptr,
                [] __host__ __device__ (T val) { return static_cast<U>(val); });

            switch(option) {
                case INCLUSIVE:
                thrust::inclusive_scan(thrust::cuda::par(*get_thrust_allocator()).on(stream), itr_in, itr_in + n, out_ptr);
                break;
                case EXCLUSIVE:
                thrust::exclusive_scan(thrust::cuda::par(*get_thrust_allocator()).on(stream), itr_in, itr_in + n, out_ptr);
                break;
                case INCLUSIVE_PAD_ZERO:
                cudaMemsetAsync(out_ptr, 0, sizeof(U), stream);
                thrust::inclusive_scan(thrust::cuda::par(*get_thrust_allocator()).on(stream), itr_in, itr_in + n, out_ptr + 1);
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
                thrust::cuda::par(*get_thrust_allocator()).on(stream), 
                itr, 
                itr + input.size(), 
                static_cast<gamax::size_type>(0),
                thrust::plus<gamax::size_type>());
        }
    };
}

    std::unique_ptr<gamax::column> gather(
        gamax::column_view gather_map, 
        gamax::column_view input, 
        cudaStream_t stream)
    {
        auto output = std::make_unique<gamax::column>(input.type(), gather_map.size());
        double_type_dispatcher(input.type(), gather_map.type(), gather_impl{}, gather_map, input, output->mutable_view(), stream);
        return output;
    }

    std::unique_ptr<gamax::column> shuffle(
        gamax::column_view input,
        const int seed,
        cudaStream_t stream)
    {
        auto output = std::make_unique<gamax::column>(input.type(), input.size());
        type_dispatcher(input.type(), shuffle_impl{}, input, output->mutable_view(), seed, stream);
        return output;
    }

    std::unique_ptr<gamax::column> compact( 
        gamax::column_view input, 
        gamax::column_view mask, 
        cudaStream_t stream)
    {
        auto output_size = thrust::reduce(
            thrust::cuda::par(*get_thrust_allocator()).on(stream), 
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

        auto res = std::make_unique<gamax::table>();

        for(gamax::size_type i = 0; i < input.num_columns(); i++) {
            if(i == 0) {
                res->add_column(compact(input[i], mask, stream));
            } else {
                res->add_column(compact(input[i], mask, res->num_rows(), stream));
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
        auto output = gamax::make_column(output_type, output_size, false, stream);

        double_type_dispatcher(input.type(), output_type, prefix_sum_impl{}, input, output->mutable_view(), option, stream);
        return output;
    }

    gamax::size_type count_nonzero(
        gamax::column_view input,
        cudaStream_t stream)
    {
        return type_dispatcher(input.type(), count_nonzero_impl{}, input, stream);
    }

} // namespace device
} // namespace util
} // namespace gamax