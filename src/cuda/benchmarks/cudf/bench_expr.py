import pylibcudf as plc
import pyarrow as pa
import timeit
from bench_cudf_utils import *
import argparse

WARMUP_RUNS = 5
BENCHMARK_RUNS = 10

def expr_factory(expr_name: str, type: pa.DataType):
    if expr_name == "one_col_mult":
        expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.MUL,
            plc.expressions.ColumnReference(0),
            plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(3, type=type))),
        )
        return expr
    elif expr_name == "one_col_exp":
        expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.DIV,
            plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(1, type=type))),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.ADD,
                plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(1, type=type))),
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.EXP,
                    plc.expressions.Operation(
                        plc.expressions.ASTOperator.MUL,
                        plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(-1, type=type))),
                        plc.expressions.ColumnReference(0),
                    ),
                ),
            ),
        )
        return expr
    elif expr_name == "two_col_arith":
        expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.MUL,
            plc.expressions.ColumnReference(0),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.SUB,
                plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(1, type=type))),
                plc.expressions.ColumnReference(1),
            ),
        )
        return expr
    elif expr_name == "three_col_arith":
        expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.MUL,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.MUL,
                plc.expressions.ColumnReference(0),
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.SUB,
                    plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(1, type=type))),
                    plc.expressions.ColumnReference(1),
                ),
            ),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.ADD,
                plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(1, type=type))),
                plc.expressions.ColumnReference(2),
            ),
        )
        return expr
    elif expr_name == "four_col_arith":
        expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.SUB,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.MUL,
                plc.expressions.ColumnReference(0),
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.SUB,
                    plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(1, type=type))),
                    plc.expressions.ColumnReference(1),
                ),
            ),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.MUL,
                plc.expressions.ColumnReference(2),
                plc.expressions.ColumnReference(3),
            ),
        )
        return expr
    elif expr_name == "simple_gt":
        expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.GREATER,
            plc.expressions.ColumnReference(0),
            plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(134217728, type=type))),
        )
        return expr
    elif expr_name == "two_col_and":
        expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.LOGICAL_AND,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.GREATER,
                plc.expressions.ColumnReference(0),
                plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(161061273, type=type))),
            ),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.LESS,
                plc.expressions.ColumnReference(1),
                plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(375809638, type=type))),
            ),
        )
        return expr
    elif expr_name == "two_col_and_or":
        expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.LOGICAL_OR,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.LOGICAL_AND,
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.EQUAL,
                    plc.expressions.ColumnReference(0),
                    plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(0, type=type))),
                ),
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.EQUAL,
                    plc.expressions.ColumnReference(1),
                    plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(1, type=type))),
                ),
            ),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.LOGICAL_AND,
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.EQUAL,
                    plc.expressions.ColumnReference(0),
                    plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(1, type=type))),
                ),
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.EQUAL,
                    plc.expressions.ColumnReference(1),
                    plc.expressions.Literal(plc.Scalar.from_arrow(pa.scalar(0, type=type))),
                ),
            ),
        )
        return expr

def generate_plc_col(num_rows, pa_type):
    init_scalar = plc.Scalar.from_arrow(pa.scalar(0, type=pa_type))
    step_scalar = plc.Scalar.from_arrow(pa.scalar(1, type=pa_type))
    result = plc.filling.sequence(
        num_rows,
        init_scalar,
        step_scalar,
    )
    return result

def generate_expr_table(num_rows, num_cols, pa_type):
    columns = []
    for _ in range(num_cols):
        col = generate_plc_col(num_rows, pa_type)
        columns.append(col)
    plc_table = plc.Table(columns)
    return plc_table

def bench_expr(plc_table: plc.Table, expr):
    # expr = plc.expressions.to_expression(expr_str, tuple([f"col{i}" for i in range(plc_table.num_columns())]))
    start_time = timeit.default_timer()
    result = plc.transform.compute_column(plc_table, expr)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def bench(plc_table: plc.Table, expr):
    results = []
    
    # warm up
    for _ in range(WARMUP_RUNS):
        _, elapsed_time = bench_expr(plc_table, expr)

    # Benchmark
    for _ in range(BENCHMARK_RUNS):
        _, elapsed_time = bench_expr(plc_table, expr)
        results.append(elapsed_time)
    
    avg_time = sum(results) / len(results)
    num_rows = plc_table.num_rows()
    thpt = num_rows / avg_time
    return avg_time, thpt

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark cudf expression evaluation")
    # parser.add_argument("--rows", type=int, default=2**29, help="Number of rows in the table")
    # parser.add_argument("--cols", type=int, default=4, help="Number of columns in the table")
    parser.add_argument("--expr", type=str, nargs='+', default=[], help="Expressions to evaluate")
    parser.add_argument("--profile", action="store_true", help="use profile mode")
    return parser.parse_args()
     
if __name__ == "__main__":
    args = parse_args()
    
    if args.profile:
        WARMUP_RUNS = 0
        BENCHMARK_RUNS = 1

    all_results = []
    
    expressions = [
        ("one_col_mult", "col0 * 3", 1),
        # ("one_col_exp", "1 / (1 + exp(-col0))", 1),
        ("two_col_arith", "col0 * (1 - col1)", 2),
        ("three_col_arith", "col0 * (1 - col1) * (1 + col2)", 3),
        ("four_col_arith", "col0 * (1 - col1) - col2 * col3", 4),
        ("simple_gt", "col0 > 134217728", 1),
        ("two_col_and", "col0 > 161061273 and col1 < 375809638", 2),
        ("two_col_and_or", "(col0 == 0 and col1 == 1) or (col0 == 1 and col1 == 0)", 2),
    ]
    
    for type in ['I32', 'I64']:
        pa_type = pa.int32() if type == 'I32' else pa.int64()
        for num_rows in [2**29]:
            for name, expr_str, num_cols in expressions:
                if len(args.expr) > 0 and name not in args.expr:
                    continue
                plc_table = generate_expr_table(num_rows, num_cols, pa_type)
                expr = expr_factory(name, pa_type)
                avg_time, thpt = bench(plc_table, expr)
                result_line = f"{type},cudf,{name},{num_rows},{num_cols},{avg_time},{thpt}"
                all_results.append(result_line)
                print(f"Type: {type}, Expr: {expr_str}, Rows: {num_rows}, Cols: {num_cols}, Avg Time: {avg_time*1000:.6f} ms, Throughput: {thpt/1_000_000_000:.2f} billion rows/s")
    
    header = "KeyType,impl,expr_type,num_items,num_cols,avg_time_s,throughput_rows_per_s"
    write_results_to_csv(all_results, "cudf_expr", header)