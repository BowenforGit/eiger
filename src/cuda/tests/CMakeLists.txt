# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG v1.15.2
    OPTIONS
        "BUILD_GMOCK OFF"
)

enable_testing()
include(GoogleTest) # Include CMake module to use gtest_discover_tests

set(testSources
    groupby/groupby_sort_test.cu
    groupby/groupby_hash_test.cu
    groupby/groupby_multikeys_test.cu
    groupby/groupby_string_test.cu
    compress/dictionary_encode_dict_ready_test.cu
    aggregate/aggregate_test.cu
    sort/sort_partition_test.cu
    memory/table_test.cu
    memory/mem_manager_test.cu
    ir/schema_test.cu
    ir/expr_test.cu
    ir/plan_node_test.cu
    ir/builder_test.cu
    ir/lower_expr_test.cu
    ir/emit_test.cu
    ir/tpch_ir_test.cu
    ir/print_test.cu
    ir/tpch_emit_test.cu
    ir/liveness_test.cu
    ir/rules_test.cu
    phys/phys_test.cu
    phys/phys_distinct_test.cu
    phys/phys_concat_test.cu
    exec/exec_test.cu
    test_header.cpp
)

list(APPEND testSources
    groupby/groupby_partition_test.cu
    compress/dictionary_encode_test.cu
    join/anti_join_test.cu
    join/ieq_join_test.cu
    join/semi_join_test.cu
    select/select_fast_test.cu
    select/predicate_test.cu
    expr/expr_fast_test.cu
    expr/expr_parser_test.cu
    types/floating_point_test.cu
    string/string_factory_test.cu
    string/string_test.cu
    string/string_contains_multiple_test.cu
    string/string_in_test.cu
    string/string_substring_test.cu
    string/string_sort_test.cu
    sort/sort_table_test.cu
    sort/topk_table_test.cu
    stats/stats_compute_test.cu
)

set(testHelperFiles
    test_helper.cu
)
    
foreach(source IN LISTS testSources)
    string(REGEX REPLACE ".*/(.*)_test.cu" "test_\\1" target ${source})
    message(STATUS "Adding test target: ${target} with source: ${source}")
    add_executable(${target} ${source} ${testHelperFiles})
    target_link_libraries(${target} PRIVATE common)
    target_link_libraries(${target} PRIVATE GTest::gtest_main)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PRIVATE gamax)
    gtest_discover_tests(${target}
        DISCOVERY_TIMEOUT 180
        PROPERTIES RESOURCE_LOCK gamax_cuda_gpu)
endforeach()

# Keep this focused binary independent of the large gamax shared library. On
# GH200/ARM, Compute Sanitizer 2025.3 can initialize CUDA while loading a
# shared object containing the library's many CUDA fatbins, before its injector
# attaches. Compiling the allocator implementation directly provides a stable,
# instrumentable direct/tracked-mode gate for its stream-lifetime contracts.
add_executable(test_allocator_sanitizer
    memory/allocator_sanitizer_test.cu
    ../memory/mem_manager.cu)
target_link_libraries(test_allocator_sanitizer PRIVATE
    common
    GTest::gtest_main
    rmm::rmm
    CCCL::CCCL
    CUDA::cuda_driver)
gtest_discover_tests(test_allocator_sanitizer
    DISCOVERY_TIMEOUT 180
    PROPERTIES RESOURCE_LOCK gamax_cuda_gpu)
