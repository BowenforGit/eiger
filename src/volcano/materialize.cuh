#pragma once

#include <cuda.h>
#include <cub/cub.cuh> 
#include <thrust/gather.h>
#include <thrust/device_ptr.h>

#include "tuple.cuh"
#include "utils.cuh"
#include "partition_util.cuh"

template<typename TupleR, 
         typename TupleS,
         typename TupleOut,
         typename idx_t>
void gfur_materialize(TupleR& r, TupleS& s, TupleOut& out, 
                      idx_t* r_match_idx, idx_t* s_match_idx, const long n_matches) {
    for_<r.num_cols-1>([&](auto i) {
        using val_t = std::tuple_element_t<i.value+1, typename TupleR::value_type>;
        thrust::device_ptr<val_t> dev_data_ptr(COL(r, i.value+1));
        thrust::device_ptr<idx_t> dev_idx_ptr(r_match_idx);
        thrust::device_ptr<val_t> dev_out_ptr(COL(out, i.value+1));
        thrust::gather(dev_idx_ptr, dev_idx_ptr+n_matches, dev_data_ptr, dev_out_ptr);
    });
    for_<s.num_cols-1>([&](auto i) {
        constexpr auto k = i.value+r.num_cols;
        using val_t = std::tuple_element_t<i.value+1, typename TupleS::value_type>;
        thrust::device_ptr<val_t> dev_data_ptr(COL(s, i.value+1));
        thrust::device_ptr<idx_t> dev_idx_ptr(s_match_idx);
        thrust::device_ptr<val_t> dev_out_ptr(COL(out, k));
        thrust::gather(dev_idx_ptr, dev_idx_ptr+n_matches, dev_data_ptr, dev_out_ptr);
    });
}

template<typename TupleR, 
         typename TupleS,
         typename TupleOut,
         typename idx_t>
void id_cluster_materialize(TupleR& r, TupleS& s, TupleOut& out, 
                            idx_t* r_idx, idx_t* s_idx, const long n_matches,
                            const int first_bit, const int radix_bits) {
    idx_t *r_idx_out, *s_idx_out;
    allocate_mem(&r_idx_out, false, n_matches*sizeof(idx_t));
    allocate_mem(&s_idx_out, false, n_matches*sizeof(idx_t));

    SinglePassPartition<idx_t, idx_t, idx_t> ssp(r_idx, s_idx, r_idx_out, s_idx_out, nullptr, n_matches, first_bit, radix_bits);
    ssp.process();

    gfur_materialize(r, s, out, r_idx_out, s_idx_out, n_matches);

    release_mem(r_idx_out);
    release_mem(s_idx_out);
}