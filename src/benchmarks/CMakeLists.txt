set(benchmarkSources 
    groupby/groupby_nvbench.cu
    compress/dict_enc.cu)

set(benchmarkTargets 
    benchmark_groupby
    benchmark_dict_enc)

foreach(source target IN ZIP_LISTS benchmarkSources benchmarkTargets)
    message(STATUS "Adding test target: ${target} with source: ${source}")
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE common)
    target_link_libraries(${target} PRIVATE nvbench::main)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PRIVATE gamax)
endforeach()

# target_link_libraries(benchmark_murmur3_hash PRIVATE murmur3)