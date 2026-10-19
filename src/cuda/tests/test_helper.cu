// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "test_helper.hpp"

#include <memory>
#include <thrust/mismatch.h>
#include <thrust/mismatch.h>
#include <thrust/execution_policy.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "types/type_dispatcher.hpp"
#include "operators/sort/sort.hpp"

namespace gamax::test {
namespace {
    struct check_cols_eq_helper {
        template<typename T>
        bool operator()(gamax::column_view c1, gamax::column_view c2) {
            auto policy = gamax::rmm_sync_exec_policy_handle{cudaStreamDefault};
            auto res = thrust::mismatch(
                policy.get(),
                c1.data<T>(),
                c1.data<T>() + c1.size(),
                c2.data<T>());

            return res.first == c1.data<T>() + c1.size();
        }
    };
}

    bool check_cols_eq(gamax::column_view c1, gamax::column_view c2) {
        if(c1.size() != c2.size() || c1.type() != c2.type()) {
            std::cout << "Columns have different sizes or types" << std::endl;
            return false;
        }
        return gamax::type_dispatcher(c1.type(), check_cols_eq_helper{}, c1, c2);
    }

    bool check_cols_eq_after_sort(gamax::column_view c1, gamax::column_view c2) {
        if(c1.size() != c2.size() || c1.type() != c2.type()) {
            std::cout << "Columns have different sizes or types" << std::endl;
            return false;
        }

        auto c1_sorted = gamax::sort::sort_column(c1, cudaStreamDefault);
        auto c2_sorted = gamax::sort::sort_column(c2, cudaStreamDefault);
        
        return check_cols_eq(c1_sorted->view(), c2_sorted->view());
    }
}
