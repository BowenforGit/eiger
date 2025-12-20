// This is to test if the *.hpp files can be safely included in *.cpp files that will be compiled by gcc compiler.

#include "memory/device_buffer.hpp"
#include "memory/scalar.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/sort/sort.hpp"
#include "operators/partition/partition.hpp"
#include "operators/join/inner_join.hpp"
#include "operators/join/join.hpp"
#include "operators/join/materialize.hpp"
#include "operators/join/semi_join.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/groupby_hash.hpp"
#include "operators/groupby/groupby_partition.hpp"
#include "operators/select/select.hpp"
#include "expr/expr_fast.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}