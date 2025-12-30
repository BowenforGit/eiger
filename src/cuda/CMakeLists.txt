if(USE_ROCM)
    CPMAddPackage(
        NAME hipco
        GITHUB_REPOSITORY ROCm/hipCollections
        GIT_TAG main
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
            "BUILD_EXAMPLES OFF"
    )
else(USE_CUDA)
    CPMAddPackage(
        NAME cuco
        GITHUB_REPOSITORY NVIDIA/cuCollections
        GIT_TAG dev
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
            "BUILD_EXAMPLES OFF"
    )

    CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        GIT_TAG main # Fetches the latest commit on the main branch
    )

    CPMAddPackage(
        NAME NVTX
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.0-c-cpp
        GIT_SHALLOW TRUE)
    target_link_libraries(common INTERFACE nvtx3-cpp)
endif()

set(srcs
    operators/groupby/groupby_sort.cu
    operators/groupby/groupby_hash.cu
    operators/groupby/aggregate.cu
    operators/sort/sort.cu
    operators/sort/sort_table.cu
    operators/partition/partition.cu
    compress/dictionary_encode_dict_ready.cu
    memory/column.cu
    memory/table.cu
    memory/mem_manager.cu
    data_gen/table_gen.cu
    data_gen/string_generator.cu
    utils/transform.cu
    string/string_factory.cu
    string/string_fast.cu
    string/string_sort.cu
    )
    
if(USE_CUDA)
    list(APPEND srcs
        compress/dictionary_encode.cu
        # operators/groupby/groupby_partition.cu
        operators/groupby/groupby.cu
        operators/join/materialize.cu
        operators/join/sort_merge_join.cu
        operators/join/inner_join.cu
        operators/join/semi_join.cu
        operators/join/semi_hash_join.cu
        operators/join/partition_hash_join.cu
        operators/join/hash_join.cu
        operators/select/select.cu
        operators/project/project.cu
        expr/expr_fast.cu
        expr/expr_nofuse.cu
        expr/expr_parser.cpp
        expr/expr_visualize.cpp
    )
endif()

add_library(gamax SHARED ${srcs})
target_link_libraries(gamax PRIVATE common)
target_link_libraries(gamax PRIVATE OpenMP::OpenMP_CXX)

if(USE_ROCM)
    target_link_libraries(gamax PUBLIC hipco)
else(USE_CUDA)
    target_link_libraries(gamax PUBLIC cuco)
    target_link_libraries(gamax PUBLIC CCCL::CCCL)
endif()
    
add_subdirectory(benchmarks)
add_subdirectory(tests)