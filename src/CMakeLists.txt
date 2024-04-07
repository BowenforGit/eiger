# add_executable(join_exp ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/join/join_exp.cu)
# target_link_libraries(join_exp PRIVATE common)

find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(join_exp PUBLIC OpenMP::OpenMP_CXX)
# endif()

set(srcs
    operators/groupby/groupby_sort.cu
    operators/groupby/groupby_partition.cu
    operators/sort/sort.cu
    operators/partition/partition.cu
    memory/column.cu
    memory/table.cu
    memory/mem_manager.cu
)

add_library(gamax SHARED ${srcs})
target_link_libraries(gamax PRIVATE common)

add_subdirectory(tests)
add_subdirectory(benchmarks)