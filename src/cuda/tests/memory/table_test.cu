#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "data_gen/generator.cuh"
#include "utils/transform.hpp"
#include "utils/utils.cuh"
#include "data_gen/table_gen.hpp"
#include "tests/test_helper.hpp"

namespace {
    void test_table_select_columns() {
        constexpr int ncols = 4;
        
        std::vector<std::unique_ptr<gamax::column>> cols;
        for(int i = 0; i < ncols; i++) {
            int val[] = {i, i+1, i+2, i+3};
            auto col = gamax::make_column_from_host(val, 4);
            cols.push_back(std::move(col));
        }
        
        auto table = gamax::make_table(std::move(cols));
        
        auto tv = table->view();
        auto tv2 = tv.select_columns({3, 0});
        ASSERT_TRUE(gamax::test::check_cols_eq(tv2[0], tv[3]));
        ASSERT_TRUE(gamax::test::check_cols_eq(tv2[1], tv[0]));

        auto tmv = table->mutable_view();
        auto tmv2 = tmv.select_columns({2, 1});
        ASSERT_TRUE(gamax::test::check_cols_eq(tmv2[0].view(), tmv[2].view()));
        ASSERT_TRUE(gamax::test::check_cols_eq(tmv2[1].view(), tmv[1].view()));
    }
}

TEST(TableTest, SelectColumns) {
    test_table_select_columns();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}