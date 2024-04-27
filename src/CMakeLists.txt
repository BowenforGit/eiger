find_package(OpenMP)

set(srcs
    operators/groupby/groupby_sort.cu
    operators/groupby/groupby_partition.cu
    operators/groupby/groupby_hash.cu
    operators/sort/sort.cu
    operators/partition/partition.cu
    compress/dictionary_encode.cu
    memory/column.cu
    memory/table.cu
    memory/mem_manager.cu
)

add_library(gamax SHARED ${srcs})
target_link_libraries(gamax PRIVATE common)
target_link_libraries(gamax PRIVATE cuco)

add_subdirectory(tests)
add_subdirectory(benchmarks)