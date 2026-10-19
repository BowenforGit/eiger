# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(OpenSSL REQUIRED)
find_path(RAPIDJSON_INCLUDE_DIR rapidjson/document.h REQUIRED)

message(STATUS "Arrow version: ${Arrow_VERSION}")
message(STATUS "Parquet version: ${Parquet_VERSION}")

set(tpch_srcs
    tpch_layout.cpp
    tpch_reader.cpp
    tpch.cu
    tpch_util.cu
    tpch_ops.cu
    ir_source_resolver.cpp
    verify.cpp
)

add_library(gamax_tpch SHARED ${tpch_srcs})
target_compile_features(gamax_tpch PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(gamax_tpch PUBLIC gamax)
target_link_libraries(gamax_tpch PRIVATE common)
target_link_libraries(gamax_tpch PRIVATE Arrow::arrow_shared)
target_link_libraries(gamax_tpch PRIVATE Parquet::parquet_shared)
target_link_libraries(gamax_tpch PRIVATE OpenSSL::Crypto)
target_include_directories(gamax_tpch PRIVATE "${RAPIDJSON_INCLUDE_DIR}")

# Arrow/Parquet from miniforge3 require a newer libstdc++ than the system default.
# Add miniforge3's lib to the link search path so its libstdc++ is found.
get_filename_component(_arrow_lib_dir "${Arrow_DIR}/../.." ABSOLUTE)
target_link_directories(gamax_tpch PUBLIC "${_arrow_lib_dir}")

add_executable(tpch_parquet_metadata tpch_parquet_metadata.cpp)
target_compile_features(tpch_parquet_metadata PRIVATE cxx_std_20)
target_link_libraries(tpch_parquet_metadata PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared)
target_include_directories(
    tpch_parquet_metadata PRIVATE "${RAPIDJSON_INCLUDE_DIR}")
target_link_directories(tpch_parquet_metadata PRIVATE "${_arrow_lib_dir}")

# Small end-to-end runner: build IR for a TPC-H query, lower, execute
# against live Parquet data via `IrSourceResolver`. Keeps the pipeline
# exercised on real data outside the gtest harness.
find_package(CUDAToolkit REQUIRED)

add_executable(run_tpch_ir run_tpch_ir.cpp)
target_link_libraries(run_tpch_ir PRIVATE gamax)
target_link_libraries(run_tpch_ir PRIVATE gamax_tpch)
target_link_libraries(run_tpch_ir PRIVATE common)
target_link_libraries(run_tpch_ir PRIVATE CUDA::cudart)
# --green-sms uses the CUDA driver-API green-ctx calls (cuInit / cuGreenCtxCreate
# / cuDevSmResourceSplitByCount) from benchmarks/green-ctx/green_ctx.hpp.
target_link_libraries(run_tpch_ir PRIVATE CUDA::cuda_driver)

if(TARGET gamax_alps)
    add_library(gamax_tpch_distributed STATIC
        distributed_tpch.cpp
        distributed_benchmark_support.cpp
        physical_plan_config.cpp)
    set_target_properties(gamax_tpch_distributed PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(gamax_tpch_distributed PUBLIC
        gamax
        gamax_tpch
        gamax_alps
        common)
    target_include_directories(gamax_tpch_distributed PRIVATE
        "${RAPIDJSON_INCLUDE_DIR}")

    add_executable(run_tpch_ir_mpi
        run_tpch_ir_mpi.cpp
        mpi_dist_helpers.cu)
    target_link_libraries(run_tpch_ir_mpi PRIVATE gamax)
    target_link_libraries(run_tpch_ir_mpi PRIVATE gamax_tpch)
    target_link_libraries(run_tpch_ir_mpi PRIVATE gamax_alps)
    target_link_libraries(run_tpch_ir_mpi PRIVATE gamax_tpch_distributed)
    target_link_libraries(run_tpch_ir_mpi PRIVATE common)
    target_link_libraries(run_tpch_ir_mpi PRIVATE CUDA::cudart)

    add_executable(cuda_visible_device_count cuda_visible_device_count.cpp)
    target_link_libraries(cuda_visible_device_count PRIVATE CUDA::cudart)

    add_executable(ncu_counter_probe ncu_counter_probe.cu)
    target_link_libraries(ncu_counter_probe PRIVATE CUDA::cudart nvtx3::nvtx3-cpp)

    add_executable(benchmark_tpch_ir_mpi benchmark_tpch_ir_mpi.cpp)
    target_link_libraries(benchmark_tpch_ir_mpi PRIVATE
        gamax_tpch_distributed
        gamax_tpch
        gamax_alps
        gamax
        common
        CUDA::cudart)
    find_package(Git QUIET)
    set(_gamax_git_revision_header
        "${CMAKE_CURRENT_BINARY_DIR}/generated/gamax_git_revision.hpp")
    add_custom_target(gamax_tpch_git_revision
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND ${CMAKE_COMMAND}
                -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
                -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
                -DOUTPUT_FILE=${_gamax_git_revision_header}
                -P ${PROJECT_SOURCE_DIR}/cmake/GenerateGitRevision.cmake
        BYPRODUCTS "${_gamax_git_revision_header}"
        VERBATIM)
    add_dependencies(benchmark_tpch_ir_mpi
        cuda_visible_device_count
        ncu_counter_probe
        gamax_tpch_git_revision)
    target_include_directories(benchmark_tpch_ir_mpi PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_sources(benchmark_tpch_ir_mpi PRIVATE
        "${_gamax_git_revision_header}")

    add_executable(synthesize_tpch_ir_mpi synthesize_tpch_ir_mpi.cpp)
    target_link_libraries(synthesize_tpch_ir_mpi PRIVATE
        gamax_tpch_distributed
        gamax_tpch
        gamax_alps
        gamax
        common
        CUDA::cudart)
else()
    message(STATUS "run_tpch_ir_mpi disabled because BUILD_ALPS is OFF")
endif()

if(BUILD_TEST)
    add_executable(tpch_reader_policy_test tests/tpch_reader_policy_test.cpp)
    target_link_libraries(tpch_reader_policy_test PRIVATE
        gamax_tpch
        gamax
        common
        GTest::gtest_main
        Arrow::arrow_shared
        Parquet::parquet_shared
        CUDA::cudart)
    gtest_discover_tests(tpch_reader_policy_test
        DISCOVERY_TIMEOUT 180
        PROPERTIES RESOURCE_LOCK gamax_cuda_gpu)

    if(TARGET gamax_tpch_distributed)
        add_executable(distributed_benchmark_support_test
            tests/distributed_benchmark_support_test.cpp)
        target_link_libraries(distributed_benchmark_support_test PRIVATE
            gamax_tpch_distributed
            GTest::gtest_main)
        gtest_discover_tests(distributed_benchmark_support_test
            DISCOVERY_TIMEOUT 180)

        add_executable(physical_plan_config_test
            tests/physical_plan_config_test.cpp)
        target_link_libraries(physical_plan_config_test PRIVATE
            gamax_tpch_distributed
            GTest::gtest_main)
        gtest_discover_tests(physical_plan_config_test
            DISCOVERY_TIMEOUT 180)

        add_test(NAME distributed_tpch_benchmark_help
            COMMAND $<TARGET_FILE:benchmark_tpch_ir_mpi> --help)
        add_test(NAME distributed_tpch_benchmark_query_help
            COMMAND $<TARGET_FILE:benchmark_tpch_ir_mpi> 6 --help)
        add_test(NAME distributed_tpch_visible_device_count_empty
            COMMAND $<TARGET_FILE:cuda_visible_device_count>
                    --print-visible-device-count)
        add_test(NAME distributed_tpch_visible_device_count_invalid_tail
            COMMAND $<TARGET_FILE:cuda_visible_device_count>
                    --print-visible-device-count)
        set_tests_properties(
            distributed_tpch_benchmark_help
            distributed_tpch_benchmark_query_help
            PROPERTIES PASS_REGULAR_EXPRESSION "usage:.*QUERY_ID")
        set_tests_properties(distributed_tpch_visible_device_count_empty
            PROPERTIES
                ENVIRONMENT "CUDA_VISIBLE_DEVICES="
                PASS_REGULAR_EXPRESSION "^0")
        set_tests_properties(distributed_tpch_visible_device_count_invalid_tail
            PROPERTIES
                ENVIRONMENT "CUDA_VISIBLE_DEVICES=0,2147483647"
                PASS_REGULAR_EXPRESSION "^1")

        add_test(NAME distributed_tpch_visible_gpu_count
            COMMAND bash
                    ${PROJECT_SOURCE_DIR}/script/tests/distributed_tpch_common_test.sh
                    ${PROJECT_SOURCE_DIR}/script/distributed_tpch_common.sh)

        add_executable(distributed_benchmark_mpi_test
            tests/distributed_benchmark_mpi_test.cpp)
        target_link_libraries(distributed_benchmark_mpi_test PRIVATE
            gamax_tpch_distributed)
        foreach(np 2 3 4)
            add_test(NAME distributed_benchmark_mpi_np${np}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                        ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:distributed_benchmark_mpi_test>
                        ${MPIEXEC_POSTFLAGS})
        endforeach()
    endif()
endif()
