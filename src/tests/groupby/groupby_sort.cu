#include <gtest/gtest.h>

#include "groupby_test.hpp"

GROUPBY_TEST_CASE_SUITE(GROUPBY_SORT, gamax::groupby::groupby_algo::GROUPBY_SORT)

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}