# GaMax CUDA Public Interfaces

This note summarizes public-facing declarations in `src/cuda/**/*.hpp`.

## Types
- [src/cuda/types/data_type.hpp](src/cuda/types/data_type.hpp) defines `size_type` (int64), `type_id` enum (primitive, string), `data_type` wrapper with equality, and `sort_order` enum.
- [src/cuda/types/type_dispatcher.hpp](src/cuda/types/type_dispatcher.hpp) provides `type_to_id`, `id_to_type_impl`, `type_dispatcher`/`type_dispatcher_all` for single-type dispatch, `double_type_dispatcher` for two-type dispatch, and helpers `size_of`, `to_string`.
- [src/cuda/types/vec_type.hpp](src/cuda/types/vec_type.hpp) maps scalar types and widths to CUDA vector types via `vec_type` and `vec_type_t`; `widest_int_for_alignment` gives the widest integer type for a given alignment.
- [src/cuda/types/sentinel.hpp](src/cuda/types/sentinel.hpp) exposes `sentinel<T>()` returning min for signed types or max for unsigned types.

## Memory primitives
- [src/cuda/memory/device_buffer.hpp](src/cuda/memory/device_buffer.hpp) RAII wrapper owning device memory; move-only, holds pointer and byte size.
- [src/cuda/memory/mem_manager.hpp](src/cuda/memory/mem_manager.hpp) declares `MemoryManager` interface and implementations `UserSpaceMM`, `CudaMM`; `ThrustAllocator` adaptor; helpers `allocate_mem`, `release_mem`, `release_all_mem`, `debug_device_mem_usage`, `get_mm`, `get_thrust_allocator`.
- [src/cuda/memory/column.hpp](src/cuda/memory/column.hpp) defines `column_view_base`, `column_view`, `mutable_column_view` for typed access; owning `column` (move-only) with buffer management; factories `make_column`, `make_idx_column`, `make_column_from_host/device`, `copy_column_to_host`, `make_sequence`, `make_column_from_view`, `free_column`.
- [src/cuda/memory/table_device_view.hpp](src/cuda/memory/table_device_view.hpp) declares experimental `table_device_view_base` (device-side view of column views) and host-side `table_view_base` with `device_view`, `select_columns`; type aliases `table_view`, `mutable_table_view`, and owning `table` with `make_table` factory.
- [src/cuda/memory/table.hpp](src/cuda/memory/table.hpp) alternative `table_device_view` for copying `table` to device; includes `create` helper.
- [src/cuda/memory/scalar.hpp](src/cuda/memory/scalar.hpp) `scalar` union wrapper storing one typed value with setters/getters and `type()` accessor.

## Utilities
- [src/cuda/utils/check_error.hpp](src/cuda/utils/check_error.hpp) debug-only CUDA error checking macros `CHECK_CUDA_ERROR`, `CHECK_LAST_CUDA_ERROR`.
- [src/cuda/utils/device_info.hpp](src/cuda/utils/device_info.hpp) compile-time constants `SM_COUNT`, `MAX_SMEM_BYTES` selected by architecture macros.
- [src/cuda/utils/transform.hpp](src/cuda/utils/transform.hpp) device utilities: `gather`, `shuffle`, `compact` (column/table variants), `prefix_sum` with options enum, and `count_nonzero`.

## Strings
- [src/cuda/string/string_column.hpp](src/cuda/string/string_column.hpp) string abstractions: `string_impl_id` enum, `string_column_view_base` (and aliases `string_column_view`, `mutable_string_column_view`), owning `string_column` (with meta buffer, view helpers), scalar views `string_scalar_view`, `string_scalar`, factories `make_string_scalar`, `make_string_column` (host/device buffers, offsets).
- [src/cuda/string/string.hpp](src/cuda/string/string.hpp) string ops with `string_op_options`: predicates `equal`, `startswith`, `endswith`, `contains`, `contains_multiple`; ordering `argsort_strings`, `sort_strings`; `gather_strings`.

## Data generation
- [src/cuda/data_gen/table_gen.hpp](src/cuda/data_gen/table_gen.hpp) table generators in `gamax::datagen::device`: `generate_table`, `pk_uniform_id`, `fk_uniform`, `fk_uniform_id`, `fk_zipf`, `fk_zipf_id` for primary/foreign keys with distributions and match ratios.
- [src/cuda/data_gen/string_generator.hpp](src/cuda/data_gen/string_generator.hpp) string generators with `string_generation_options`, enums `string_length_distribution`, `pattern_position`; APIs `generate_strings` and `generate_random_strings`.

## Compression
- [src/cuda/compress/dictionary_encode.hpp](src/cuda/compress/dictionary_encode.hpp) dictionary encoding overloads for columns/tables with optional prebuilt dictionary and encode type; returns encoded column (and dictionary size when generated).

## Expression evaluation
- [src/cuda/expr/expr_fast.hpp](src/cuda/expr/expr_fast.hpp) expression IR: `ExprId`, `Expr`, `ExprProgram`, `ExprTree` with factory methods (constants, column refs, unary/binary/ternary ops), operator overloads, `make_program` helpers, and type utilities.
- [src/cuda/expr/expr_parser.hpp](src/cuda/expr/expr_parser.hpp) AST structures (`AstNodeKind`, `AstUnaryOp`, `AstBinaryOp`, `AstNode`) and functions `parse_expression_to_ast`, `ast_to_expr_tree`, `parse_expression_to_tree`.

## Operators
- GroupBy
  - [src/cuda/operators/groupby/aggregation.hpp](src/cuda/operators/groupby/aggregation.hpp) aggregation enums (`aggregation_id`), `aggregation_request`, `agg_output_data_type`, `aggregate` helper, and `DISPATCH_AGGREGATE` macro.
  - [src/cuda/operators/groupby/groupby.hpp](src/cuda/operators/groupby/groupby.hpp) `groupby_algo`, `groupby_hint`, and main `groupby` entry (key/value tables plus aggregation requests and hints).
  - [src/cuda/operators/groupby/groupby_hash.hpp](src/cuda/operators/groupby/groupby_hash.hpp) hash-based `groupby` specialization.
  - [src/cuda/operators/groupby/groupby_sort.hpp](src/cuda/operators/groupby/groupby_sort.hpp) sort-based `groupby` and `groupby_gfur` variant.
  - [src/cuda/operators/groupby/groupby_partition.hpp](src/cuda/operators/groupby/groupby_partition.hpp) partition-based `groupby` and `groupby_gfur` with detailed notes.
- Sort
  - [src/cuda/operators/sort/sort.hpp](src/cuda/operators/sort/sort.hpp) column/table sorting APIs: `sort_column_pair`, `sort_column` (views and owning), `argsort`, `sort_table` overloads, `topk_table` with sort keys/orders.
- Partition
  - [src/cuda/operators/partition/partition.hpp](src/cuda/operators/partition/partition.hpp) radix partitioning: `partition_column_pair`, `partition_column`, `arg_partition`, `compute_partition_offsets` for owning columns and views.
- Project / Select
  - [src/cuda/operators/project/project.hpp](src/cuda/operators/project/project.hpp) `project_options` and `project` entry plus `project_fuse` / `project_nofuse` variants applying expression lists to a table.
  - [src/cuda/operators/select/select.hpp](src/cuda/operators/select/select.hpp) `select_options` and `select` entry plus fused/non-fused variants applying a predicate expression.
- Join
  - [src/cuda/operators/join/join.hpp](src/cuda/operators/join/join.hpp) join constants (`MAX_JOIN_OUTPUT_SIZE`), enums `join_algo`, `materialization_algo`, and `join_hint` configuration.
  - [src/cuda/operators/join/inner_join.hpp](src/cuda/operators/join/inner_join.hpp) `inner_eq_join` entry and detail implementations (`ieq_hash_join`, sort-merge, partition hash variants) plus `get_ieq_join_output_alloc_size`.
  - [src/cuda/operators/join/semi_join.hpp](src/cuda/operators/join/semi_join.hpp) `semi_join` entry with detail hash/partition variants and `get_semi_join_output_alloc_size`.
  - [src/cuda/operators/join/materialize.hpp](src/cuda/operators/join/materialize.hpp) `detail::simple_materialize` for materializing columns via index maps.

## Strings (operators)
- [src/cuda/string/string.hpp](src/cuda/string/string.hpp) also includes argsort/sort/gather specific to strings (see Strings section above).

## Miscellaneous
- [src/cuda/operators/partition/partition.hpp](src/cuda/operators/partition/partition.hpp) also exposes bit-range partitioning offsets helper.
- [src/cuda/utils/transform.hpp](src/cuda/utils/transform.hpp) includes `count_nonzero` convenience for masks.

These links point to the source for full signatures and usage details.
