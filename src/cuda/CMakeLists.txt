if(USE_ROCM)
    CPMAddPackage(
        NAME libhipcxx
        GITHUB_REPOSITORY ROCm/libhipcxx
        GIT_TAG 9ac637b66019a8fcf6796f8cfd7b2c13f76313b6 # release/2.7.x @ 2025-11-04
    )
    CPMAddPackage(
        NAME cuco
        GITHUB_REPOSITORY ROCm/hipCollections
        GIT_TAG 77f2e84e4b6f9d667d8733967301dbdad4e795de # release/rocmds-25.10 @ 2025-11-11
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
            "BUILD_EXAMPLES OFF"
    )
else(USE_NVIDIA)
    CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        GIT_TAG v3.3.2
    )

    CPMAddPackage(
        NAME cuco
        GITHUB_REPOSITORY NVIDIA/cuCollections
        GIT_TAG 6477be2182668015f9a91e3a0bb7e248eceecd09 # dev @ 2026-03-31
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
            "BUILD_EXAMPLES OFF"
    )

    CPMAddPackage(
        NAME NVTX
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.4.0-c-cpp
        GIT_SHALLOW TRUE)
    target_link_libraries(common INTERFACE nvtx3-cpp)
endif()

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
    tpch/tpch_ir.cu
    )
    
if(USE_NVIDIA)
    list(APPEND srcs
        compress/dictionary_encode.cu
        compress/multikey_compress.cu
        operators/groupby/groupby_partition.cu
        operators/groupby/groupby.cu
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
    )
endif()

add_library(gamax SHARED ${srcs})
target_link_libraries(gamax PRIVATE common)
target_link_libraries(gamax PRIVATE OpenMP::OpenMP_CXX)

if(USE_ROCM)
    target_link_libraries(gamax PUBLIC cuco::cuco)
    target_link_libraries(gamax PUBLIC libhipcxx::libhipcxx)
else(USE_NVIDIA)
    target_link_libraries(gamax PUBLIC cuco)
    target_link_libraries(gamax PUBLIC CCCL::CCCL)
endif()
    
add_subdirectory(benchmarks)
add_subdirectory(tests)

if(BUILD_TPCH AND USE_NVIDIA)
    add_subdirectory(tpch)
endif()