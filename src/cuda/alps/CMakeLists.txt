# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# ALPS — multi-GPU execution component (optional, BUILD_ALPS).
#
# Builds a shared library `gamax_alps` (shuffle + distributed join), an MPI
# benchmark executable, and MPI correctness test executables. Requires NCCL
# and MPI in addition to the base gamax dependencies.

find_package(MPI REQUIRED)
find_package(NCCL REQUIRED)
find_package(CUDAToolkit REQUIRED)

message(STATUS "ALPS: MPI ${MPI_CXX_VERSION} (${MPIEXEC_EXECUTABLE})")
message(STATUS "ALPS: NCCL ${NCCL_VERSION} (${NCCL_LIBRARY})")

set(alps_srcs
    context.cpp
    exec/distributed_executor.cpp
    exec/distributed_planner.cpp
    collectives/alltoall.cu
    collectives/alltoall_ce.cu
    collectives/alltoall_p2p.cu
    collectives/allgather.cu
    collectives/allgather_ce.cu
    collectives/allgather_p2p.cu
    collectives/ce_pull_shuffle.cu
    collectives/fabric_scratch.cu
    collectives/shuffle.cu
    collectives/table_exchange.cu
    containers/fabric_region.cu
    containers/fabric_peer_buffer.cu
    operators/dist_join.cu
    operators/column_overlap_eq_join.cu
    operators/distributed_relational.cu
    operators/shared_map_join.cu
    operators/local_map_join.cu
    datagen/gen_shard.cu
)

add_library(gamax_alps SHARED ${alps_srcs})
# CUDA::cuda_driver provides the VMM / fabric-handle driver API (cuMem*) used by
# the direct-P2P all-to-all in collectives/alltoall_p2p.cu.
target_link_libraries(gamax_alps PUBLIC gamax MPI::MPI_CXX NCCL::nccl CUDA::cudart CUDA::cuda_driver)
target_link_libraries(gamax_alps PRIVATE common OpenMP::OpenMP_CXX)

# Helper to wire an alps executable (bench or test) with the common deps.
function(alps_add_executable target src)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE gamax_alps gamax common
        MPI::MPI_CXX NCCL::nccl CUDA::cudart)
endfunction()

# --- Benchmarks -------------------------------------------------------------
alps_add_executable(alps_dist_join_bench         bench/dist_join_bench.cu)
alps_add_executable(alps_dist_join_scaling_bench bench/dist_join_scaling_bench.cu)
alps_add_executable(alps_shuffle_bench           bench/shuffle_bench.cu)
alps_add_executable(alps_allgather_bench         bench/allgather_bench.cu)
alps_add_executable(alps_align_bench             bench/align_bench.cu)
alps_add_executable(alps_partition_probe         bench/partition_probe.cu)
alps_add_executable(alps_join_probe              bench/join_probe.cu)
alps_add_executable(alps_shared_map_bench        bench/shared_map_bench.cu)
alps_add_executable(alps_dist_join_compare_bench bench/dist_join_compare_bench.cu)
alps_add_executable(alps_column_overlap_join_bench bench/column_overlap_join_bench.cu)
target_link_libraries(alps_shared_map_bench PRIVATE CUDA::nvml) # NVML: --nvlink-bytes counters
target_link_libraries(alps_dist_join_compare_bench PRIVATE CUDA::nvml) # NVML: --nvlink-bytes counters

# --- Tests ------------------------------------------------------------------
alps_add_executable(alps_shuffle_test    tests/shuffle_test.cu)
alps_add_executable(alps_dist_join_test  tests/dist_join_test.cu)
alps_add_executable(alps_allgather_test  tests/allgather_test.cu)
alps_add_executable(alps_table_exchange_test tests/table_exchange_test.cu)
alps_add_executable(alps_distributed_executor_test tests/distributed_executor_test.cu)
alps_add_executable(alps_shared_map_test tests/shared_map_test.cu)
alps_add_executable(alps_shared_map_join_test tests/shared_map_join_test.cu)
alps_add_executable(alps_column_overlap_join_test tests/column_overlap_join_test.cu)
alps_add_executable(alps_ce_pull_capability_test tests/ce_pull_capability_test.cu)
alps_add_executable(alps_ce_pull_shuffle_test tests/ce_pull_shuffle_test.cu)
alps_add_executable(alps_ce_pull_join_test tests/ce_pull_join_test.cu)

# Run the same test binary under different GPU counts. ALPS never maps multiple
# local ranks to one GPU; on a smaller worker, make_context emits the diagnostic
# below and CTest records a named hardware skip.
set(_alps_mpi_flags ${MPIEXEC_PREFLAGS})
# Power-of-two and non-power-of-two rank counts. np3 exercises the general hash
# range-reduction routing (rank_bits=2 -> 4 buckets, bucket 3 stays empty), not
# just the top-bits-of-pow2 fast path. Counts kept <= 4 so a 4-GPU box can run
# them without oversubscribing GPUs across NCCL ranks.
foreach(np 2 3 4)
    add_test(NAME alps_shuffle_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_shuffle_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_dist_join_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_dist_join_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_allgather_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_allgather_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_table_exchange_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_table_exchange_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_distributed_executor_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_distributed_executor_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_column_overlap_join_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_column_overlap_join_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_shared_map_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_shared_map_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_ce_pull_capability_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_ce_pull_capability_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_ce_pull_shuffle_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_ce_pull_shuffle_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME alps_ce_pull_join_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_alps_mpi_flags}
                $<TARGET_FILE:alps_ce_pull_join_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        alps_shuffle_np${np}
        alps_dist_join_np${np}
        alps_allgather_np${np}
        alps_table_exchange_np${np}
        alps_distributed_executor_np${np}
        alps_column_overlap_join_np${np}
        alps_shared_map_np${np}
        alps_ce_pull_capability_np${np}
        alps_ce_pull_shuffle_np${np}
        alps_ce_pull_join_np${np}
        PROPERTIES
            SKIP_REGULAR_EXPRESSION "requires a distinct GPU"
            RESOURCE_LOCK gamax_cuda_gpu)
    set_tests_properties(
        alps_ce_pull_capability_np${np}
        alps_ce_pull_shuffle_np${np}
        alps_ce_pull_join_np${np}
        PROPERTIES TIMEOUT 120)
endforeach()

# A rank-local phase failure must be reduced before any rank branches into the
# next collective verifier. Without that coordination this test times out.
add_test(NAME alps_ce_pull_capability_np2_asymmetric_phase_failure
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${_alps_mpi_flags}
            $<TARGET_FILE:alps_ce_pull_capability_test>
            --inject-asymmetric-failure-after-batch ${MPIEXEC_POSTFLAGS})
set_tests_properties(
    alps_ce_pull_capability_np2_asymmetric_phase_failure
    PROPERTIES
        TIMEOUT 60
        SKIP_REGULAR_EXPRESSION "requires a distinct GPU"
        RESOURCE_LOCK gamax_cuda_gpu)

# FabricPeerBuffer::ensure must reduce a rank-local allocation/export failure
# before healthy ranks enter the fabric-handle Allgather.
add_test(NAME alps_ce_pull_capability_np2_asymmetric_vmm_allocation_failure
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${_alps_mpi_flags}
            $<TARGET_FILE:alps_ce_pull_capability_test>
            --inject-asymmetric-vmm-allocation-failure ${MPIEXEC_POSTFLAGS})
set_tests_properties(
    alps_ce_pull_capability_np2_asymmetric_vmm_allocation_failure
    PROPERTIES
        TIMEOUT 60
        SKIP_REGULAR_EXPRESSION "requires a distinct GPU"
        RESOURCE_LOCK gamax_cuda_gpu)

# The default matrix above covers the low-level unordered path. Exercise both
# sender-order policies at the largest local rank count as well.
foreach(schedule rotated round-robin-waves)
    string(REPLACE "-" "_" schedule_test_name "${schedule}")
    add_test(NAME alps_ce_pull_shuffle_np4_${schedule_test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${_alps_mpi_flags}
                $<TARGET_FILE:alps_ce_pull_shuffle_test>
                --schedule ${schedule} ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
        alps_ce_pull_shuffle_np4_${schedule_test_name}
        PROPERTIES
            TIMEOUT 120
            SKIP_REGULAR_EXPRESSION "requires a distinct GPU"
            RESOURCE_LOCK gamax_cuda_gpu)
endforeach()

# VMM allocations are padded as a whole. A non-power-of-two slot count guards
# against accidentally deriving a misaligned per-slot stride from that extent.
add_test(NAME alps_ce_pull_shuffle_np4_three_slots
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${_alps_mpi_flags}
            $<TARGET_FILE:alps_ce_pull_shuffle_test>
            --schedule round-robin-waves --slots 3 ${MPIEXEC_POSTFLAGS})
set_tests_properties(
    alps_ce_pull_shuffle_np4_three_slots
    PROPERTIES
        TIMEOUT 120
        SKIP_REGULAR_EXPRESSION "requires a distinct GPU"
        RESOURCE_LOCK gamax_cuda_gpu)
