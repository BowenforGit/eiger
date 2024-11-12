#include <gtest/gtest.h>

#include "groupby_test.hpp"
#include "benchmarks/groupby/groupby_util.cuh"

GROUPBY_TEST_CASE_SUITE(GROUPBY_HASH, gamax::groupby::groupby_algo::GROUPBY_HASH)

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}