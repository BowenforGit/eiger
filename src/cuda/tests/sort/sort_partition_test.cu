#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "operators/sort/sort.hpp"
#include "operators/partition/partition.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "utils/transform.hpp"
#include "utils/utils.cuh"
#include "data_gen/table_gen.hpp"
#include "tests/test_helper.hpp"

template<typename T>
void test_partition(
    gamax::size_type nrows, 
    gamax::size_type begin_bit,
    gamax::size_type end_bit)
{
    auto type = gamax::data_type(gamax::type_to_id<T>());
    auto tin = gamax::datagen::device::fk_uniform_id(type, nrows, 2*nrows, 4, 1.0, 42);  

    auto vtin = tin->view();

    std::vector<std::unique_ptr<gamax::column>> partitioned_cols(vtin.num_columns());
    for (gamax::size_type i = 1; i < vtin.num_columns(); i++)
    {
        auto cols = gamax::partition::partition_column_pair(
            vtin[0], vtin[i], 0, 5);

        partitioned_cols[i] = std::move(std::get<1>(cols));
        if (i == 1)
        {
            partitioned_cols[0] = std::move(std::get<0>(cols));
        }

        ASSERT_EQ(partitioned_cols[i]->size(), 2*nrows);
        ASSERT_TRUE(gamax::test::check_cols_eq_after_sort(partitioned_cols[i]->view(), vtin[i]));
        ASSERT_TRUE(gamax::test::check_cols_eq(partitioned_cols[0]->view(), partitioned_cols[i]->view()));
    }
}

TEST(PartitionTest, Sum32) {
    test_partition<int32_t>(4864, 0, 5);
}

TEST(PartitionTest, Max32) {
    test_partition<int32_t>(4865, 0, 5);
}

TEST(PartitionTest, Sum64) {
    test_partition<int64_t>(4864, 0, 5);
}

TEST(PartitionTest, Max64) {
    test_partition<int64_t>(4865, 0, 5);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}