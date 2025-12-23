if(USE_ROCM)
    CPMAddPackage(
        NAME hipbench
        GITHUB_REPOSITORY ROCm/hipBench
        GIT_TAG main
        OPTIONS
            "NVBench_ENABLE_TESTING OFF"
    )
else(USE_CUDA)
    CPMAddPackage(
        NAME nvbench
        GITHUB_REPOSITORY NVIDIA/nvbench
        GIT_TAG main
    )
endif()

set(benchmarkSources 
    groupby/groupby_nvbench.cu
    compress/dict_enc_nvbench.cu
    join/ieq_join_nvbench.cu
    containers/multimap_nvbench.cu
    containers/map_nvbench.cu
    sort/sort_nvbench.cu
    select/select_fast_nvbench.cu
    select/select_handwritten_nvbench.cu
    string/string_nvbench.cu
    expr/expr_nvbench.cu
)

foreach(source IN LISTS benchmarkSources)
    string(REGEX REPLACE ".*/(.*)_nvbench.cu" "benchmark_\\1" target ${source})
    message(STATUS "Adding benchmark target: ${target} with source: ${source}")
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE common)
    target_link_libraries(${target} PRIVATE nvbench::main)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PRIVATE gamax)
endforeach()
