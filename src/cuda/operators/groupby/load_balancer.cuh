#pragma once

#include <cuda.h>
#include <cub/cub.cuh>

#include "utils/utils.cuh"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax {
namespace groupby {

template<typename NumItemsT = gamax::size_type, typename IdxT = NumItemsT>
__global__ void generate_workunits_kernel(const IdxT* membership_ps, 
                        const NumItemsT num_groups,
                        const IdxT threshold,
                        NumItemsT* num_units,
                        IdxT* units_start, 
                        IdxT* units_end, 
                        IdxT* units_group) {
    for(gamax::size_type i = get_cuda_tid(); i < num_groups; i += nthreads()) {
        IdxT start = (i == 0) ? 0 : membership_ps[i-1];
        IdxT end = membership_ps[i];
        IdxT num_items = end - start;
        IdxT num_work_units = (num_items+threshold-1)/threshold;
        for(gamax::size_type j = 0; j < num_work_units; j++) {
            IdxT unit_start = j*threshold;
            IdxT unit_end = min((j+1)*threshold, num_items);
            gamax::size_type idx = atomicAdd(num_units, 1);
            units_start[idx] = start+unit_start;
            units_end[idx] = start+unit_end;
            units_group[idx] = i;
        }
    }
}

template<typename NumItemsT = gamax::size_type, typename IdxT = NumItemsT>
void generate_workunits(const IdxT* membership_ps, 
                        const NumItemsT num_groups,
                        const IdxT threshold,
                        NumItemsT* num_units,
                        IdxT* units_start, 
                        IdxT* units_end, 
                        IdxT* units_group,
                        cudaStream_t stream = cudaStreamDefault) {
    NumItemsT* d_num_units = nullptr;
    allocate_mem(&d_num_units, true, sizeof(NumItemsT));
    generate_workunits_kernel<<<num_tb(num_groups, 1024), 1024, 0, stream>>>(membership_ps, num_groups, threshold, d_num_units, units_start, units_end, units_group);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(num_units, d_num_units, sizeof(NumItemsT), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    release_mem(d_num_units); 
}
} // namespace groupby
} // namespace gamax