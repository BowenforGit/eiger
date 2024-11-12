#pragma once

#include "volcano/join_base.hpp"
#include "volcano/sort_merge_join.cuh"
#include "volcano/partitioned_hash_join.cuh"
#include "volcano/sort_hash_join.cuh

#include "nvbench/nvbench.cuh"

enum class AvailJoinAlgo {
    SMJ_GFUR,
    SMJ_GFTR,
    PHJ_GFUR,
    SHJ_GFUR,
    SHJ_IC,
    SHJ_GFTR,
};

template<typename TupleR, typename TupleS, typename ResultTuple>
auto get_join_impl(TupleR r, TupleS s, int circular_buffer_size, nvbench::state &state) {
    // partition related parameters for the partitioned hash join
    auto const first_bit = state.get_int32("first_bit");
    auto const log_part1 = state.get_int32("log_part1");
    auto const log_part2 = state.get_int32("log_part2");

    JoinBase<ResultTuple>* impl = nullptr;
    switch (state.get_string("join_algo")) {
        case "SMJ_GFUR":
            impl = new GFUR_Join<TupleR, TupleS, ResultTuple>(r, s, circular_buffer_size, JoinAlgo::SMJ, MaterializeAlgo::GFUR);
            break;
        case "SMJ_GFTR":
            impl = new SortMergeJoin<TupleR, TupleS, ResultTuple, true>(r, s, circular_buffer_size);
            break;
        case "PHJ_GFUR":
            impl = new PartitionHashJoin<TupleR, TupleS, ResultTuple>(r, s, log_part1, log_part2, first_bit, circular_buffer_size);
            break;
        case "SHJ_GFUR":
            impl = new GFUR_Join<TupleR, TupleS, ResultTuple>(r, s, circular_buffer_size, JoinAlgo::SHJ, MaterializeAlgo::GFUR);
            break;
        case "SHJ_IC":
            impl = new GFUR_Join<TupleR, TupleS, ResultTuple>(r, s, circular_buffer_size, JoinAlgo::SHJ, MaterializeAlgo::ID_CLUSTER);
            break;
        case "SHJ_GFTR":
            impl = new SortHashJoin<TupleR, TupleS, ResultTuple>(r, s, first_bit, log_part1+log_part2, circular_buffer_size);
            break;
        default:
            throw std::runtime_error("Unknown join algorithm");
    }
    return impl;
}