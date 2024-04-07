enable_testing()
include(GoogleTest)

set(testSources 
    groupby/groupby_sort.cu
    groupby/groupby_partition.cu
)

set(testTargets 
    test_groupby_sort
    test_groupby_partition
)
    
foreach(source target IN ZIP_LISTS testSources testTargets)
    message(STATUS "Adding test target: ${target} with source: ${source}")
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE common)
    target_link_libraries(${target} PRIVATE GTest::gtest_main)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PRIVATE gamax)
    gtest_discover_tests(${target})
endforeach()
