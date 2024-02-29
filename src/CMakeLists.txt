add_executable(join_exp ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/join/join_exp.cu)
target_link_libraries(join_exp PRIVATE common)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(join_exp PUBLIC OpenMP::OpenMP_CXX)
endif()

add_subdirectory(tests)
add_subdirectory(benchmarks)