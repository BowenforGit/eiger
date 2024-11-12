#include "transform.hpp"

#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#include <memory>

#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

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

} // namespace device
} // namespace util
} // namespace gamax