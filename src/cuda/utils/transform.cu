#include "transform.hpp"

#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#include <cub/device/device_select.cuh>
#include <memory>
#include <stdexcept>

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
        template <typename T>
        void operator()(
            gamax::column_view const& gather_map, 
            gamax::column_view const& input, 
            gamax::mutable_column_view output, 
            cudaStream_t stream)
        {
            const auto N = gather_map.size();

            thrust::device_ptr<const gamax::size_type> gather_map_ptr(gather_map.data<gamax::size_type>());
            thrust::device_ptr<const T> input_ptr(input.data<T>());
            thrust::device_ptr<T> output_ptr(output.data<T>());

            thrust::gather(thrust::device.on(stream), gather_map_ptr, gather_map_ptr + N, input_ptr, output_ptr);
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
}

    std::unique_ptr<gamax::column> gather(
        gamax::column_view gather_map, 
        gamax::column_view input, 
        cudaStream_t stream)
    {
        auto output = std::make_unique<gamax::column>(input.type(), gather_map.size());
        type_dispatcher(input.type(), gather_impl{}, gather_map, input, output->mutable_view(), stream);
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

} // namespace device
} // namespace util
} // namespace gamax