if(USE_ROCM)
    CPMAddPackage(
        NAME hipbench
        GITHUB_REPOSITORY ROCm/hipBench
        GIT_TAG main
        OPTIONS
            "NVBench_ENABLE_TESTING OFF"
    )
    set(GAMAX_BENCHMARK_MAIN_TARGET nvbench::main)
    if(TARGET hipbench::main)
        set(GAMAX_BENCHMARK_MAIN_TARGET hipbench::main)
    endif()
else(USE_NVIDIA)
    CPMAddPackage(
        NAME nvbench
        GITHUB_REPOSITORY NVIDIA/nvbench
        GIT_TAG main
    )
    set(GAMAX_BENCHMARK_MAIN_TARGET nvbench::main)
endif()

set(benchmarkSources 
    groupby/groupby_nvbench.cu
    compress/dict_enc_nvbench.cu
    join/ieq_join_nvbench.cu
    containers/multimap_nvbench.cu
    containers/map_nvbench.cu
    sort/sort_nvbench.cu
    sort/sort_table_nvbench.cu
    select/select_fast_nvbench.cu
    select/select_handwritten_nvbench.cu
    string/string_nvbench.cu
    string/string_sort_nvbench.cu
    expr/expr_nvbench.cu
    misc/hash_fn_nvbench.cu
    misc/gather_nvbench.cu
    misc/read_write_nvbench.cu
    stats/stats_nvbench.cu
)

foreach(source IN LISTS benchmarkSources)
    string(REGEX REPLACE ".*/(.*)_nvbench.cu" "benchmark_\\1" target ${source})
    message(STATUS "Adding benchmark target: ${target} with source: ${source}")
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE common)
    target_link_libraries(${target} PRIVATE ${GAMAX_BENCHMARK_MAIN_TARGET})
    if(USE_ROCM)
        target_compile_definitions(${target} PRIVATE GAMAX_DISABLE_NVTX)
    endif()
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PRIVATE gamax)
endforeach()
