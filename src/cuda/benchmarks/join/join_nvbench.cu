#define CUB_STDERR
// #define CHECK_CORRECTNESS
// #define SORTED_REL
#define MR_FILTER_FK

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <unistd.h>
#include <fstream>

#include <cuda.h>

#include "tuple.cuh"
#include "utils.cuh"
#include "operators.cuh"
#include "../data_gen/generator.cuh"
#include "sort_merge_join.cuh"
#include "partitioned_hash_join.cuh"
#include "sort_hash_join.cuh"
#include "experiment_util.cuh"
#include "join_base.hpp"
#include "gfur_join.cuh"

#include "nvbench/nvbench.cuh"
#include "join_nvbench_util.cuh"


NVBENCH_BENCH(join_benchmark)
    .add_int64_axis("Duration (us)", nvbench::range(0, 100, 5))
    .set_timeout(1); // Limit to one second per measurement.