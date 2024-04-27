#pragma once

#include <cuda.h>
#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <iterator>

#include "utils/utils.cuh"

namespace gamax {
namespace util {
namespace device {
    template<typename MapIt,
            typename InputIt,
            typename OutputIt,
            typename NumItemsT,
            typename Map = typename std::iterator_traits<MapIt>::value_type,
            typename Input = typename std::iterator_traits<InputIt>::value_type,
            typename Output = typename std::iterator_traits<OutputIt>::value_type>
    void gather(MapIt gather_map, InputIt input, OutputIt output, NumItemsT N, cudaStream_t stream = cudaStreamDefault) {
        thrust::device_ptr<Map> gather_map_ptr(gather_map);
        thrust::device_ptr<Input> input_ptr(input);
        thrust::device_ptr<Output> output_ptr(output);

        thrust::gather(thrust::device.on(stream), gather_map_ptr, gather_map_ptr + N, input_ptr, output_ptr);
    }

    template<typename MapIt,
            typename InputIt,
            typename OutputIt,
            typename NumItemsT,
            typename Map = typename std::iterator_traits<MapIt>::value_type,
            typename Input = typename std::iterator_traits<InputIt>::value_type,
            typename Output = typename std::iterator_traits<OutputIt>::value_type>
    void scatter(MapIt scatter_map, InputIt input, OutputIt output, NumItemsT N, cudaStream_t stream = cudaStreamDefault) {
        thrust::device_ptr<Map> scatter_map_ptr(scatter_map);
        thrust::device_ptr<Input> input_ptr(input);
        thrust::device_ptr<Output> output_ptr(output);

        thrust::scatter(thrust::device.on(stream), input_ptr, input_ptr + N, scatter_map_ptr, output_ptr);
    }

    /**
     * output[i] = lookup_map[input[i]]
    */
    template<typename MapIt,
            typename InputIt,
            typename OutputIt,
            typename NumItemsT>
    void lookup(MapIt lookup_map, InputIt input, OutputIt output, NumItemsT N, cudaStream_t stream = cudaStreamDefault) {
        gather(input, lookup_map, output, N, stream);
    }
}
}
}