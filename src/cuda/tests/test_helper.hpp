#pragma once

#include <memory>
#include <thrust/mismatch.h>
#include <thrust/mismatch.h>
#include <thrust/execution_policy.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/type_dispatcher.hpp"

namespace gamax::test {
    bool check_cols_eq(gamax::column_view c1, gamax::column_view c2);
    bool check_cols_eq_after_sort(gamax::column_view c1, gamax::column_view c2);
}