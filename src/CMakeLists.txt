add_executable(join_nvbench ${PROJECT_SOURCE_DIR}/src/benchmarks/join_nvbench.cu)
target_compile_features(join_nvbench PUBLIC cxx_std_17 cuda_std_17)

set(COMPILE_FLAGS "")
list(APPEND COMPILE_FLAGS -gencode=arch=compute_86,code=\"sm_86,compute_86\")
list(APPEND COMPILE_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --extended-lambda)
list(APPEND COMPILE_FLAGS -Xptxas="-v" -lineinfo -Xcudafe)
list(APPEND COMPILE_FLAGS -Xcompiler -fopenmp -O3 --compile)
list(APPEND COMPILE_FLAGS -Xlinker -lgomp)

target_compile_options(
    join_nvbench
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${COMPILE_FLAGS}>
)

target_compile_definitions(
    join_nvbench
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:SM860>
)

target_include_directories(
    join_nvbench 
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/"
           "${PROJECT_SOURCE_DIR}/nvbench/"
)

target_link_libraries(join_nvbench nvbench::main)