#include <gtest/gtest.h>

#include "groupby_test.hpp"

GROUPBY_TEST_CASE_SUITE(GROUPBY_PARTITION, gamax::groupby::groupby_algo::GROUPBY_PARTITION)

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}