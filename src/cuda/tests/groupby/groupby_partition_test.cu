#include <gtest/gtest.h>

#include "groupby_test.hpp"

GROUPBY_TEST_CASE_SUITE(GROUPBY_PARTITION, gamax::groupby::groupby_algo::PARTITION)
GROUPBY_TEST_CASE_SUITE(GROUPBY_PARTITION_GFUR, gamax::groupby::groupby_algo::PARTITION_GFUR)

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}