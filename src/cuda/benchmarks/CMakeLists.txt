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
)

set(benchmarkTargets 
    benchmark_groupby
    benchmark_dict_enc
)

foreach(source target IN ZIP_LISTS benchmarkSources benchmarkTargets)
    message(STATUS "Adding test target: ${target} with source: ${source}")
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE common)
    target_link_libraries(${target} PRIVATE nvbench::main)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PRIVATE gamax)
endforeach()
