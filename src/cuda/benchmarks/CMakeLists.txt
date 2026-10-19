# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

CPMAddPackage(
    NAME nvbench
    GITHUB_REPOSITORY NVIDIA/nvbench
    GIT_TAG b0a46f44c21c344e2c338225d724eed60d912fe2 # main @ 2026-04-14
    OPTIONS
        "NVBench_ENABLE_CUPTI OFF"
)
set(GAMAX_BENCHMARK_MAIN_TARGET nvbench::main)

set(benchmarkSources 
    groupby/groupby_nvbench.cu
    compress/dict_enc_nvbench.cu
    join/ieq_join_nvbench.cu
    # containers/multimap_nvbench.cu
    containers/multiset_nvbench.cu
    containers/multiset_chunked_nvbench.cu
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
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PRIVATE gamax)
endforeach()

# Host-side allocator overhead/scaling guard. This is intentionally a plain
# executable rather than an NVBench GPU-kernel benchmark.
add_executable(benchmark_allocator_latency memory/allocator_latency_bench.cu)
target_link_libraries(benchmark_allocator_latency PRIVATE common gamax)

# Generic two-stream complete-operation concurrency benchmark. This is a
# standalone harness (rather than NVBench) because it owns two independently
# timed streams, paired baselines, direct sequential controls, and adaptive
# per-cell sampling.
add_executable(benchmark_concurrency
    concurrency/benchmark_concurrency.cu
    concurrency/operations.cu
    concurrency/timing.cu
    concurrency/reporting.cpp)
target_link_libraries(benchmark_concurrency PRIVATE common gamax)
if(BUILD_TEST)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_test(NAME concurrency_gate_self_test
        COMMAND benchmark_concurrency --self-test-gate)
    set_tests_properties(concurrency_gate_self_test PROPERTIES
        RESOURCE_LOCK gamax_cuda_gpu
        TIMEOUT 30)

    add_test(NAME concurrency_expr_gather_smoke
        COMMAND benchmark_concurrency
            --op-a expr.two_col_add.nofuse
            --op-b misc.gather.thrust
            --a-exp-min 12 --a-exp-max 12
            --b-exp-min 12 --b-exp-max 12
            --warmups 0 --min-samples 1 --max-samples 1
            --bootstrap-iterations 100
            --output-prefix ${CMAKE_CURRENT_BINARY_DIR}/concurrency_expr_gather_smoke)
    set_tests_properties(concurrency_expr_gather_smoke PROPERTIES
        RESOURCE_LOCK gamax_cuda_gpu
        TIMEOUT 30)

    add_test(NAME concurrency_multiset_smoke
        COMMAND benchmark_concurrency
            --op-a containers.multiset.insert
            --op-b containers.multiset.retrieve
            --a-exp-min 12 --a-exp-max 12
            --b-exp-min 12 --b-exp-max 12
            --warmups 0 --min-samples 1 --max-samples 1
            --bootstrap-iterations 100
            --output-prefix ${CMAKE_CURRENT_BINARY_DIR}/concurrency_multiset_smoke)
    set_tests_properties(concurrency_multiset_smoke PROPERTIES
        RESOURCE_LOCK gamax_cuda_gpu
        TIMEOUT 30)

    add_test(NAME concurrency_chunked_expr_gather_smoke
        COMMAND benchmark_concurrency
            --op-a expr.two_col_add.nofuse
            --op-b misc.gather.thrust
            --a-exp-min 12 --a-exp-max 12
            --b-exp-min 12 --b-exp-max 12
            --a-chunk-size 256 --b-chunk-size 256
            --warmups 0 --min-samples 1 --max-samples 1
            --bootstrap-iterations 100
            --output-prefix ${CMAKE_CURRENT_BINARY_DIR}/concurrency_chunked_expr_gather_smoke)
    set_tests_properties(concurrency_chunked_expr_gather_smoke PROPERTIES
        RESOURCE_LOCK gamax_cuda_gpu
        TIMEOUT 30)

    add_test(NAME concurrency_chunked_multiset_smoke
        COMMAND benchmark_concurrency
            --op-a containers.multiset.insert
            --op-b containers.multiset.retrieve
            --a-exp-min 12 --a-exp-max 12
            --b-exp-min 12 --b-exp-max 12
            --a-chunk-size 256 --b-chunk-size 256
            --warmups 0 --min-samples 1 --max-samples 1
            --bootstrap-iterations 100
            --output-prefix ${CMAKE_CURRENT_BINARY_DIR}/concurrency_chunked_multiset_smoke)
    set_tests_properties(concurrency_chunked_multiset_smoke PROPERTIES
        RESOURCE_LOCK gamax_cuda_gpu
        TIMEOUT 30)

    add_test(NAME concurrency_chunked_expr_unaligned_rejected
        COMMAND benchmark_concurrency
            --op-a expr.two_col_add.nofuse
            --op-b expr.two_col_add.nofuse
            --a-exp-min 12 --a-exp-max 12
            --b-exp-min 12 --b-exp-max 12
            --a-chunk-size 255 --b-chunk-size 255
            --warmups 0 --min-samples 1 --max-samples 1
            --bootstrap-iterations 100
            --output-prefix ${CMAKE_CURRENT_BINARY_DIR}/concurrency_chunked_expr_unaligned_rejected)
    set_tests_properties(concurrency_chunked_expr_unaligned_rejected PROPERTIES
        RESOURCE_LOCK gamax_cuda_gpu
        TIMEOUT 30
        PASS_REGULAR_EXPRESSION
            "expression vector kernel requires 16-byte-aligned input and output addresses")

    add_test(NAME concurrency_chunked_summary_regeneration
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/concurrency/check_concurrency_reporting.py
            regenerate
            ${CMAKE_CURRENT_SOURCE_DIR}/concurrency/summarize_concurrency.py
            ${CMAKE_CURRENT_BINARY_DIR}/concurrency_chunked_expr_gather_smoke_raw.csv
            ${CMAKE_CURRENT_BINARY_DIR}/concurrency_chunked_expr_gather_smoke_metadata.json
            ${CMAKE_CURRENT_BINARY_DIR}/concurrency_chunked_expr_gather_smoke_summary.csv)
    set_tests_properties(concurrency_chunked_summary_regeneration PROPERTIES
        DEPENDS concurrency_chunked_expr_gather_smoke
        TIMEOUT 30)

    add_test(NAME concurrency_failure_summary_smoke
        COMMAND benchmark_concurrency
            --op-a expr.two_col_add.nofuse
            --op-b expr.two_col_add.nofuse
            --a-exp-min 33 --a-exp-max 33
            --b-exp-min 33 --b-exp-max 33
            --warmups 0 --min-samples 1 --max-samples 1
            --bootstrap-iterations 100
            --output-prefix ${CMAKE_CURRENT_BINARY_DIR}/concurrency_failure_summary_smoke)
    set_tests_properties(concurrency_failure_summary_smoke PROPERTIES
        RESOURCE_LOCK gamax_cuda_gpu
        TIMEOUT 30)

    add_test(NAME concurrency_failure_summary_schema
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/concurrency/check_concurrency_reporting.py
            width
            ${CMAKE_CURRENT_BINARY_DIR}/concurrency_failure_summary_smoke_summary.csv)
    set_tests_properties(concurrency_failure_summary_schema PROPERTIES
        DEPENDS concurrency_failure_summary_smoke
        TIMEOUT 30)
endif()

# Green-context SM-scaling benchmark variants. Near-duplicates of the originals
# extended with a `green_sms` axis that routes the timed op onto a CUDA
# green-context SM partition. They call the driver API directly (cuInit / green
# ctx / SM split) so they additionally link CUDA::cuda_driver.
if(TRUE)
    find_package(CUDAToolkit REQUIRED)  # provides CUDA::cuda_driver (libcuda, green-ctx API)
    set(greenBenchmarkSources
        green-ctx/multiset_green_nvbench.cu
        green-ctx/gather_green_nvbench.cu
        green-ctx/ieq_join_green_nvbench.cu
        green-ctx/sort_green_nvbench.cu
    )
    foreach(source IN LISTS greenBenchmarkSources)
        string(REGEX REPLACE ".*/(.*)_nvbench.cu" "benchmark_\\1" target ${source})
        message(STATUS "Adding green-ctx benchmark target: ${target} with source: ${source}")
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE common)
        target_link_libraries(${target} PRIVATE ${GAMAX_BENCHMARK_MAIN_TARGET})
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(${target} PRIVATE gamax)
        target_link_libraries(${target} PRIVATE CUDA::cuda_driver)
    endforeach()
endif()

# TPC-H benchmark: needs gamax_tpch (which links gamax publicly)
if(BUILD_TPCH)
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

    # Green-context SM-scaling variant of the IR TPC-H bench: adds a
    # `green_sms` axis that routes the whole query onto an N-SM green-ctx
    # partition (query-level SM-budget study). Lowers with the canonical
    # default config only (no operator-algo axes) so the timed config
    # matches run_tpch_ir --green-sms --verify. Driver-API green calls →
    # additionally links CUDA::cuda_driver (find_package above in the
    # CUDA toolkit discovery above provides the CUDA:: imported targets).
    add_executable(benchmark_tpch_ir_green green-ctx/tpch_ir_green_nvbench.cu)
    target_link_libraries(benchmark_tpch_ir_green PRIVATE common)
    target_link_libraries(benchmark_tpch_ir_green PRIVATE ${GAMAX_BENCHMARK_MAIN_TARGET})
    target_link_libraries(benchmark_tpch_ir_green PRIVATE gamax_tpch)
    target_link_libraries(benchmark_tpch_ir_green PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(benchmark_tpch_ir_green PRIVATE CUDA::cuda_driver)

    # IR-driven TPC-H axis probe — fast failure-mode coverage tool.
    # Bypasses NVBench's measurement loop and runs each (query × axis)
    # cell once with try/catch, emitting a CSV of pass/fail outcomes.
    add_executable(tpch_ir_axis_probe tpch/tpch_ir_axis_probe.cu)
    target_link_libraries(tpch_ir_axis_probe PRIVATE common)
    target_link_libraries(tpch_ir_axis_probe PRIVATE gamax_tpch)
    target_link_libraries(tpch_ir_axis_probe PRIVATE OpenMP::OpenMP_CXX)

    find_package(Threads REQUIRED)
    # Two-query TPC-H concurrency benchmark. Standalone (not NVBench): owns two
    # host threads driving independent streams, a host barrier for simultaneous
    # start, adaptive per-cell sampling, and NVML clock/throttle recording
    # (clock pinning is unavailable to non-root on this box).
    add_executable(benchmark_tpch_concurrency
        tpch_concurrency/benchmark_tpch_concurrency.cu)
    target_compile_features(benchmark_tpch_concurrency PRIVATE cxx_std_20 cuda_std_20)
    target_link_libraries(benchmark_tpch_concurrency PRIVATE common)
    target_link_libraries(benchmark_tpch_concurrency PRIVATE gamax_tpch)
    target_link_libraries(benchmark_tpch_concurrency PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(benchmark_tpch_concurrency PRIVATE Threads::Threads)
    target_link_libraries(benchmark_tpch_concurrency PRIVATE CUDA::cudart)
    target_link_libraries(benchmark_tpch_concurrency PRIVATE CUDA::nvml)
    # Green-context SM partitioning (green_ctx.hpp) uses the CUDA driver API
    # (cuGreenCtxCreate / cuDevSmResourceSplitByCount / cuGreenCtxStreamCreate).
    target_link_libraries(benchmark_tpch_concurrency PRIVATE CUDA::cuda_driver)
endif()
# Target-specific options live in a separate file so that editing them
# does not cause CMake to re-evaluate CPMAddPackage in this file.
include(${CMAKE_CURRENT_LIST_DIR}/benchmark_options.cmake)
