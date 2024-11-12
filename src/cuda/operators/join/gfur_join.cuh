#pragma once
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <tuple>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 

#include "types/tuple.cuh"
#include "utils/utils.cuh"
#include "sort_hash_join.cuh"
#include "sort_merge_join.cuh"
#include "join_base.hpp"
#include "materialize.cuh"

template<typename TupleR,
         typename TupleS,
         typename TupleOut>
class GFUR_Join : public JoinBase<TupleOut> {
    static_assert(TupleR::num_cols >= 2 && 
                  TupleS::num_cols >= 2 && 
                  TupleR::num_cols+TupleS::num_cols == TupleOut::num_cols+1);
public:
    explicit GFUR_Join(TupleR r_in, TupleS s_in, int circular_buffer_size, JoinAlgo join_algo, MaterializeAlgo mat_algo = MaterializeAlgo::GFUR) 
             : r(r_in)
             , s(s_in)
             , circular_buffer_size(circular_buffer_size)
             , mat_algo(mat_algo) {
        nr = r.num_items;
        ns = s.num_items;

        allocate_mem(&r_idx, false, sizeof(int)*nr);
        allocate_mem(&s_idx, false, sizeof(int)*ns);
        fill_sequence<<<num_tb(nr), 1024>>>((int*)(r_idx), 0, nr);
        fill_sequence<<<num_tb(ns), 1024>>>((int*)(s_idx), 0, ns);

        struct Chunk<key_t, int> temp_r;
        temp_r.add_column(COL(r,0));
        temp_r.add_column(r_idx);
        temp_r.num_items = nr;
        
        struct Chunk<key_t, int> temp_s;
        temp_s.add_column(COL(s,0));
        temp_s.add_column(s_idx);
        temp_s.num_items = ns;

        if(join_algo == JoinAlgo::SHJ) {
            const int radix_bits = (std::max(nr, ns) < (1<<28)) ? 15 : 16;
            joiner = new SortHashJoin<decltype(temp_r), decltype(temp_s), struct Chunk<key_t, int, int>>(temp_r, temp_s, 0, radix_bits, circular_buffer_size);
        } else if(join_algo == JoinAlgo::SMJ) {
            joiner = new SortMergeJoin<decltype(temp_r), decltype(temp_s), struct Chunk<key_t, int, int>>(temp_r, temp_s, circular_buffer_size);
        } else {
            std::cout << "Join algorithm not supported.\n";
            std::abort();
        }

        cudaEventCreate(&start); 
        cudaEventCreate(&stop);
    }

    ~GFUR_Join() {
        release_mem(r_idx);
        release_mem(s_idx);
        delete joiner;
    }

public:
    TupleOut join() override {
        auto temp_out = joiner->join();
        transform_time = joiner->all_stats()[0];
        mf_time = joiner->all_stats()[1];
        
        n_matches = temp_out.num_items;
        r_match_idx = temp_out.template get_typed_ptr<1>();
        s_match_idx = temp_out.template get_typed_ptr<2>();
        out.allocate(circular_buffer_size);
        cudaMemcpy(COL(out, 0), COL(temp_out, 0), sizeof(key_t)*circular_buffer_size, cudaMemcpyDeviceToDevice);
        out.num_items = n_matches;
        
        TIME_FUNC_ACC(materialize(), mat_time);
        temp_out.free_mem();

        return out;
    }
    void print_stats() override {
        std::cout << "Sort: " << transform_time << " ms\n"
                  << "Merge: " << mf_time << " ms\n"
                  << "Materialize: " << mat_time << " ms\n\n";
    }

    std::vector<float> all_stats() override {
        return {transform_time, mf_time, mat_time};
    }

public:
    float transform_time {0};
    float mf_time {0};
    float mat_time {0};

private:
    void materialize() {
        if(mat_algo == MaterializeAlgo::GFUR) {
            gfur_materialize(r, s, out, r_match_idx, s_match_idx, std::min(circular_buffer_size, n_matches));
        } else if(mat_algo == MaterializeAlgo::ID_CLUSTER) {
            id_cluster_materialize(r, s, out, r_match_idx, s_match_idx, std::min(circular_buffer_size, n_matches), 16, 16);
        } else {
            std::cout << "Materialization algorithm not supported.\n";
        }
    }

private:
    TupleR r;
    TupleS s;
    TupleOut out;
    
    int nr;
    int ns;
    int n_matches;
    int circular_buffer_size;

    JoinAlgo join_algo;
    MaterializeAlgo mat_algo;

    using key_t = std::tuple_element_t<0, typename TupleR::value_type>;

    int*   r_idx         {nullptr};
    int*   s_idx         {nullptr};
    int*   r_match_idx   {nullptr};
    int*   s_match_idx   {nullptr};

    static constexpr auto  r_cols = TupleR::num_cols;
    static constexpr auto  s_cols = TupleS::num_cols;

    JoinBase<struct Chunk<key_t, int, int>>* joiner {nullptr};

    cudaEvent_t start;
    cudaEvent_t stop;
};