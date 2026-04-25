if(USE_ROCM)
    CPMAddPackage(
        NAME hipbench
        GITHUB_REPOSITORY ROCm/hipBench
        GIT_TAG 5a7d2b452feaa9d32d194dc989f494ea6afa13b3 # main @ 2024-10-30
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
        GIT_TAG b0a46f44c21c344e2c338225d724eed60d912fe2 # main @ 2026-04-14
        OPTIONS
            "NVBench_ENABLE_CUPTI OFF"
    )
    set(GAMAX_BENCHMARK_MAIN_TARGET nvbench::main)
endif()

set(benchmarkSources 
    groupby/groupby_nvbench.cu
    compress/dict_enc_nvbench.cu
    join/ieq_join_nvbench.cu
    # containers/multimap_nvbench.cu
    containers/multiset_nvbench.cu
    containers/map_nvbench.cu
    sort/sort_nvbench.cu
    sort/sort_table_nvbench.cu
    sort/sort_table_fused_key_nvbench.cu
    select/select_fast_nvbench.cu
    select/select_handwritten_nvbench.cu
    string/string_nvbench.cu
    string/string_sort_nvbench.cu
    expr/expr_nvbench.cu
    misc/hash_fn_nvbench.cu
    misc/gather_nvbench.cu
    misc/read_write_nvbench.cu
    misc/ilp_nvbench.cu
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

# TPC-H benchmark: needs gamax_tpch (which links gamax publicly)
if(BUILD_TPCH AND USE_NVIDIA)
    add_executable(benchmark_tpch tpch/tpch_nvbench.cu)
    target_link_libraries(benchmark_tpch PRIVATE common)
    target_link_libraries(benchmark_tpch PRIVATE ${GAMAX_BENCHMARK_MAIN_TARGET})
    target_link_libraries(benchmark_tpch PRIVATE gamax_tpch)
    target_link_libraries(benchmark_tpch PRIVATE OpenMP::OpenMP_CXX)

    # IR-driven TPC-H benchmark — uses the IR/ExecPlan pipeline plus
    # NVBench's manual-timer mode to exclude SourceTable resolution
    # (Parquet decode + H2D) from the timed region.
    add_executable(benchmark_tpch_ir tpch/tpch_ir_nvbench.cu)
    target_link_libraries(benchmark_tpch_ir PRIVATE common)
    target_link_libraries(benchmark_tpch_ir PRIVATE ${GAMAX_BENCHMARK_MAIN_TARGET})
    target_link_libraries(benchmark_tpch_ir PRIVATE gamax_tpch)
    target_link_libraries(benchmark_tpch_ir PRIVATE OpenMP::OpenMP_CXX)

    # IR-driven TPC-H axis probe — fast failure-mode coverage tool.
    # Bypasses NVBench's measurement loop and runs each (query × axis)
    # cell once with try/catch, emitting a CSV of pass/fail outcomes.
    add_executable(tpch_ir_axis_probe tpch/tpch_ir_axis_probe.cu)
    target_link_libraries(tpch_ir_axis_probe PRIVATE common)
    target_link_libraries(tpch_ir_axis_probe PRIVATE gamax_tpch)
    target_link_libraries(tpch_ir_axis_probe PRIVATE OpenMP::OpenMP_CXX)
endif()
# Target-specific options live in a separate file so that editing them
# does not cause CMake to re-evaluate CPMAddPackage in this file.
include(${CMAKE_CURRENT_LIST_DIR}/benchmark_options.cmake)
