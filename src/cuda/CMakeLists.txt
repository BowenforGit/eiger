# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# Resolve the complete RAPIDS 26.06 dependency cohort from the RAPIDS table
# initialized at the repository root.  In particular, this pins CCCL to
# 207502e... and cuCollections to f517bbb....
include(${rapids-cmake-dir}/cpm/cccl.cmake)
rapids_cpm_cccl()
include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco()
# RMM 26.06 is a compiled library.  Fetch the exact release tag and keep its
# tests/benchmarks out of Eiger's dependency build.
CPMAddPackage(
    NAME rmm
    GITHUB_REPOSITORY rapidsai/rmm
    GIT_TAG v26.06.00
    SOURCE_SUBDIR cpp
    OPTIONS
        "BUILD_TESTS OFF"
        "BUILD_BENCHMARKS OFF"
        "RMM_NVTX OFF"
)

target_link_libraries(common INTERFACE nvtx3::nvtx3-cpp)

set(srcs
    operators/groupby/groupby_sort.cu
    operators/groupby/groupby_hash.cu
    operators/groupby/aggregate.cu
    operators/sort/sort.cu
    operators/sort/sort_table.cu
    operators/sort/sort_table_fused_key.cu
    operators/partition/partition.cu
    compress/dictionary_encode_dict_ready.cu
    memory/column.cu
    memory/table.cu
    memory/mem_manager.cu
    data_gen/table_gen.cu
    data_gen/string_generator.cu
    utils/copying.cu
    utils/transform.cu
    utils/debug_print.cu
    stats/stats_compute.cu
    string/string_factory.cu
    string/string_fast.cu
    string/string_sort.cu
    ir/column_id.cpp
    ir/catalog.cpp
    ir/expr.cpp
    ir/plan_node.cpp
    ir/builder.cpp
    ir/lower_expr.cpp
    ir/emit.cpp
    ir/print.cpp
    ir/liveness.cpp
    ir/rules.cpp
    ir/lower_to_exec_plan.cpp
    exec/executor.cpp
    exec/print.cpp
    exec/step_profile.cpp
    tpch/tpch_ir.cu
    compress/dictionary_encode.cu
    compress/multikey_compress.cu
    operators/groupby/groupby_partition.cu
    operators/groupby/groupby.cu
    operators/join/anti_join.cu
    operators/join/cross_join.cu
    operators/join/materialize.cu
    operators/join/sort_merge_join.cu
    operators/join/inner_join.cu
    operators/join/semi_join.cu
    operators/join/semi_hash_join.cu
    operators/join/partition_hash_join.cu
    operators/join/hash_join.cu
    operators/select/select.cu
    operators/select/predicate.cu
    operators/project/project.cu
    operators/sort/topk_table.cu
    expr/expr_fast.cu
    expr/expr_nofuse.cu
    expr/expr_parser.cpp
    expr/expr_visualize.cpp
    phys/phys.cu
    )

add_library(gamax SHARED ${srcs})
target_link_libraries(gamax PRIVATE common)
target_link_libraries(gamax PRIVATE OpenMP::OpenMP_CXX)

# Opt-in debug macro for the phys-layer tracer wired in
# `src/cuda/exec/executor.cpp`. When `-DGAMAX_PHYS_DEBUG=ON` is passed
# to cmake, the executor compiles a per-step `print_table` hook gated
# at runtime by the `GAMAX_PHYS_DEBUG` env var (comma-separated
# StepKind names or `all`). Off by default — zero overhead for
# production builds.
option(GAMAX_PHYS_DEBUG "Compile the per-step phys-layer debug printer" OFF)
if(GAMAX_PHYS_DEBUG)
    target_compile_definitions(gamax PUBLIC GAMAX_PHYS_DEBUG)
endif()

target_link_libraries(gamax PUBLIC cuco::cuco CCCL::CCCL rmm::rmm)
target_link_libraries(gamax PRIVATE CUDA::cuda_driver)
    
if(BUILD_ALPS)
    add_subdirectory(alps)
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmarks)
endif()

if(BUILD_TEST)
    add_subdirectory(tests)
endif()

if(BUILD_TPCH)
    add_subdirectory(tpch)
endif()
