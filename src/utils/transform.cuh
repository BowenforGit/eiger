#pragma once

#include <cuda.h>
#include <thrust/gather.h>
#include <thrust/device_ptr.h>

#include "utils/utils.cuh"

namespace gamax {
namespace util {
namespace device {
    template<typename MapIt,
            typename InputIt,
            typename OutputIt,
            typename NumItemsT>
    void gather(MapIt gather_map, InputIt input, OutputIt output, NumItemsT N) {
        thrust::device_ptr<MapIt> gather_map_ptr(gather_map);
        thrust::device_ptr<InputIt> input_ptr(input);
        thrust::device_ptr<OutputIt> output_ptr(output);

        thrust::gather(gather_map_ptr, gather_map_ptr + N, input_ptr, output_ptr);
    }

    template<typename MapIt,
            typename InputIt,
            typename OutputIt,
            typename NumItemsT>
    void scatter(MapIt scatter_map, InputIt input, OutputIt output, NumItemsT N) {
        thrust::device_ptr<MapIt> scatter_map_ptr(scatter_map);
        thrust::device_ptr<InputIt> input_ptr(input);
        thrust::device_ptr<OutputIt> output_ptr(output);

        thrust::scatter(input_ptr, input_ptr + N, scatter_map_ptr, output_ptr);
    }

    /**
     * output[i] = lookup_map[input[i]]
    */
    template<typename MapIt,
            typename InputIt,
            typename OutputIt,
            typename NumItemsT>
    void lookup(MapIt lookup_map, InputIt input, OutputIt output, NumItemsT N) {
        gather(input, lookup_map, output, N);
    }
}
}
}