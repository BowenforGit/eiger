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
    compress/dictionary_encode_dict_ready_test.cu
    aggregate/aggregate_test.cu
    sort/sort_partition_test.cu
    memory/table_test.cu
    ir/schema_test.cu
    ir/expr_test.cu
    ir/plan_node_test.cu
    ir/builder_test.cu
    ir/lower_expr_test.cu
    ir/emit_test.cu
    ir/tpch_ir_test.cu
    ir/print_test.cu
    test_header.cpp
)

if(USE_NVIDIA)
    list(APPEND testSources
        groupby/groupby_partition_test.cu
        compress/dictionary_encode_test.cu
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
        string/string_sort_test.cu
        sort/sort_table_test.cu
        sort/topk_table_test.cu
        stats/stats_compute_test.cu
    )
endif()

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
    gtest_discover_tests(${target} PROPERTIES DISCOVERY_TIMEOUT 180)
endforeach()
