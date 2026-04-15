find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

message(STATUS "Arrow version: ${Arrow_VERSION}")
message(STATUS "Parquet version: ${Parquet_VERSION}")

set(tpch_srcs
    tpch_reader.cpp
    tpch.cu
    tpch_util.cu
    tpch_ops.cu
)

add_library(gamax_tpch SHARED ${tpch_srcs})
target_link_libraries(gamax_tpch PUBLIC gamax)
target_link_libraries(gamax_tpch PRIVATE common)
target_link_libraries(gamax_tpch PRIVATE Arrow::arrow_shared)
target_link_libraries(gamax_tpch PRIVATE Parquet::parquet_shared)

# Arrow/Parquet from miniforge3 require a newer libstdc++ than the system default.
# Add miniforge3's lib to the link search path so its libstdc++ is found.
get_filename_component(_arrow_lib_dir "${Arrow_DIR}/../.." ABSOLUTE)
target_link_directories(gamax_tpch PUBLIC "${_arrow_lib_dir}")
