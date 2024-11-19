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
    # groupby/groupby_partition_test.cu
    groupby/groupby_hash_test.cu
    # compress/dictionary_encode_test.cu
    compress/dictionary_encode_dict_ready_test.cu
    # join/ieq_join_test.cu
    aggregate/aggregate_test.cu
    sort/sort_partition_test.cu
)

set(testTargets 
    test_groupby_sort
    # test_groupby_partition
    test_groupby_hash
    # test_dictionary_encode
    test_dictionary_encode_dict_ready
    # test_ieq_join
    test_aggregate
    test_sort_partition
)

set(testHelperFiles
    test_helper.cu
)
    
foreach(source target IN ZIP_LISTS testSources testTargets)
    message(STATUS "Adding test target: ${target} with source: ${source}")
    add_executable(${target} ${source} ${testHelperFiles})
    target_link_libraries(${target} PRIVATE common)
    target_link_libraries(${target} PRIVATE GTest::gtest_main)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PRIVATE gamax)
    gtest_discover_tests(${target})
endforeach()
