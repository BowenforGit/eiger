// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "operation.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "containers/multiset.cuh"
#include "data_gen/table_gen.hpp"
#include "memory/device_buffer.hpp"
#include "memory/mem_manager.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"
#include "types/sentinel.hpp"
#include "utils/transform.hpp"

namespace gamax::benchmark::concurrency {
namespace {

__global__ void validate_expr_kernel(
    std::int64_t const* lhs, std::int64_t const* rhs, std::int64_t const* output,
    std::uint64_t size, int* mismatch)
{
    std::uint64_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint64_t const stride = blockDim.x * gridDim.x;
    for (std::uint64_t i = tid; i < size; i += stride) {
        if (output[i] != lhs[i] + rhs[i]) { atomicExch(mismatch, 1); }
    }
}

__global__ void two_col_add_stage(
    std::int64_t const* lhs, std::int64_t const* rhs,
    std::int64_t* output, std::uint64_t size)
{
    constexpr int vector_width = 4;
    std::uint64_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint64_t const stride = blockDim.x * gridDim.x;
    std::uint64_t const vector_size = (size / vector_width) * vector_width;
    for (std::uint64_t i = tid * vector_width; i < vector_size;
         i += stride * vector_width) {
        auto const a = reinterpret_cast<longlong4_16a const*>(lhs)[i / vector_width];
        auto const b = reinterpret_cast<longlong4_16a const*>(rhs)[i / vector_width];
        longlong4_16a value;
        value.x = a.x + b.x;
        value.y = a.y + b.y;
        value.z = a.z + b.z;
        value.w = a.w + b.w;
        reinterpret_cast<longlong4_16a*>(output)[i / vector_width] = value;
    }
    for (std::uint64_t i = vector_size + tid; i < size; i += stride) {
        output[i] = lhs[i] + rhs[i];
    }
}

__global__ void validate_gather_kernel(
    gamax::size_type const* map, std::int64_t const* input,
    std::int64_t const* output, std::uint64_t size, int* mismatch)
{
    std::uint64_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint64_t const stride = blockDim.x * gridDim.x;
    for (std::uint64_t i = tid; i < size; i += stride) {
        auto const index = map[i];
        if (index < 0 || static_cast<std::uint64_t>(index) >= size ||
            output[i] != input[index]) {
            atomicExch(mismatch, 1);
        }
    }
}

__global__ void validate_multiset_find_kernel(
    std::int64_t const* build_keys,
    std::int64_t const* probe_keys,
    gamax::size_type const* build_tids,
    gamax::size_type num_build_keys,
    gamax::size_type num_probe_keys,
    gamax::size_type empty_tid,
    int* mismatch)
{
    std::uint64_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint64_t const stride = blockDim.x * gridDim.x;
    for (std::uint64_t i = tid; i < static_cast<std::uint64_t>(num_probe_keys); i += stride) {
        auto const build_tid = build_tids[i];
        if (build_tid == empty_tid || build_tid < 0 || build_tid >= num_build_keys ||
            build_keys[build_tid] != probe_keys[i]) {
            atomicExch(mismatch, 1);
        }
    }
}

int validation_grid(std::uint64_t size)
{
    return static_cast<int>(std::min<std::uint64_t>((size + 255) / 256, 4096));
}

void launch_expr(
    std::int64_t const* lhs, std::int64_t const* rhs, std::int64_t* output,
    std::uint64_t size, cudaStream_t stream)
{
    constexpr std::uintptr_t vector_alignment = alignof(longlong4_16a);
    auto const aligned = [](auto const* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer) % vector_alignment == 0;
    };
    if (!aligned(lhs) || !aligned(rhs) || !aligned(output)) {
        fail("expression vector kernel requires 16-byte-aligned input and output addresses; "
             "int64 chunk boundaries must be multiples of 2 items");
    }
    constexpr int block_size = 256;
    constexpr int vector_width = 4;
    int const grid_size = static_cast<int>(
        (size + block_size * vector_width - 1) / (block_size * vector_width));
    two_col_add_stage<<<grid_size, block_size, 0, stream>>>(lhs, rhs, output, size);
    cuda_check(cudaGetLastError(), "two-column add launch");
}

class PreparedExpr final : public PreparedOperation {
public:
    explicit PreparedExpr(std::unique_ptr<gamax::table> input) : input_{std::move(input)} {}

    gamax::column_ptr enqueue(cudaStream_t stream) override
    {
        auto const size = static_cast<std::uint64_t>(input_->num_rows());
        auto output = gamax::make_column<std::int64_t>(
            static_cast<gamax::size_type>(size), stream, gamax::buffer_init::uninitialized);
        auto const table = input_->view();
        launch_expr(table[0].data<std::int64_t>(), table[1].data<std::int64_t>(),
                    output->mutable_view().data<std::int64_t>(), size, stream);
        return output;
    }

    void begin_chunked(std::uint64_t chunk_size, cudaStream_t stream) override
    {
        size_ = static_cast<std::uint64_t>(input_->num_rows());
        chunk_size_ = effective_chunk_size(size_, chunk_size);
        output_ = gamax::make_column<std::int64_t>(
            static_cast<gamax::size_type>(size_), stream, gamax::buffer_init::uninitialized);
    }

    std::uint64_t chunk_count() const override { return chunk_count_for(size_, chunk_size_); }

    void enqueue_chunk(std::uint64_t chunk_index, cudaStream_t stream) override
    {
        auto const offset = chunk_index * chunk_size_;
        auto const length = std::min(chunk_size_, size_ - offset);
        auto const table = input_->view();
        launch_expr(table[0].data<std::int64_t>() + offset,
                    table[1].data<std::int64_t>() + offset,
                    output_->mutable_view().data<std::int64_t>() + offset,
                    length, stream);
    }

    gamax::column_ptr finish_chunked() override { return std::move(output_); }

    void validate(gamax::column_ptr const& result_owner, cudaStream_t stream) const override
    {
        if (!result_owner) { fail("expression produced no result"); }
        auto const result = result_owner->view();
        if (result.type().id() != gamax::type_id::INT64 || result.size() != input_->num_rows()) {
            fail("expression result shape/type mismatch");
        }
        gamax::device_buffer mismatch(sizeof(int), stream, gamax::buffer_init::zero);
        auto const table = input_->view();
        validate_expr_kernel<<<validation_grid(result.size()), 256, 0, stream>>>(
            table[0].data<std::int64_t>(), table[1].data<std::int64_t>(),
            result.data<std::int64_t>(), result.size(), static_cast<int*>(mismatch.data()));
        cuda_check(cudaGetLastError(), "expression validation launch");
        int host_mismatch = 0;
        cuda_check(cudaMemcpyAsync(&host_mismatch, mismatch.data(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   "expression validation copy");
        cuda_check(cudaStreamSynchronize(stream), "expression validation synchronization");
        if (host_mismatch) { fail("expression correctness validation failed"); }
    }

private:
    std::unique_ptr<gamax::table> input_;
    gamax::column_ptr output_;
    std::uint64_t size_{};
    std::uint64_t chunk_size_{};
};

class ExprDescriptor final : public OperationDescriptor {
public:
    std::string stable_name() const override { return k_expr_name; }
    std::string axis_name() const override { return "num_items"; }
    std::string config_string(GenericConfig const& config) const override
    {
        return "KeyType=int64_t;impl=nofuse;expr=col0+col1;num_items=" +
               std::to_string(config.size());
    }
    void validate_config(GenericConfig const& config) const override
    {
        if (config.size_exponent < 1 || config.size_exponent > 61) {
            fail("expression size exponent must be in [1, 61]");
        }
    }
    std::size_t estimate_memory(GenericConfig const& config) const override
    {
        return static_cast<std::size_t>(config.size()) * sizeof(std::int64_t) * 3;
    }
    std::unique_ptr<PreparedOperation> prepare(
        GenericConfig const& config, cudaStream_t setup_stream) const override
    {
        auto type = gamax::data_type(gamax::type_id::INT64);
        return std::make_unique<PreparedExpr>(gamax::datagen::device::pk_uniform_id(
            type, static_cast<gamax::size_type>(config.size()), 2, k_data_seed, setup_stream));
    }
};

class PreparedGather final : public PreparedOperation {
public:
    PreparedGather(std::unique_ptr<gamax::table> input, std::unique_ptr<gamax::table> map)
        : input_{std::move(input)}, map_{std::move(map)} {}

    gamax::column_ptr enqueue(cudaStream_t stream) override
    {
        return gamax::util::device::gather(map_->view()[0], input_->view()[0], stream);
    }

    void begin_chunked(std::uint64_t chunk_size, cudaStream_t stream) override
    {
        size_ = static_cast<std::uint64_t>(map_->num_rows());
        chunk_size_ = effective_chunk_size(size_, chunk_size);
        output_ = gamax::make_column<std::int64_t>(
            static_cast<gamax::size_type>(size_), stream, gamax::buffer_init::uninitialized);
    }

    std::uint64_t chunk_count() const override { return chunk_count_for(size_, chunk_size_); }

    void enqueue_chunk(std::uint64_t chunk_index, cudaStream_t stream) override
    {
        auto const offset = chunk_index * chunk_size_;
        auto const length = std::min(chunk_size_, size_ - offset);
        auto const full_map = map_->view()[0];
        gamax::column_view map_slice(
            full_map.data<gamax::size_type>() + offset,
            static_cast<gamax::size_type>(length), full_map.type());
        gamax::util::device::gather_into(
            map_slice, input_->view()[0],
            output_->mutable_view(static_cast<gamax::size_type>(offset)), stream);
    }

    gamax::column_ptr finish_chunked() override { return std::move(output_); }

    void validate(gamax::column_ptr const& result_owner, cudaStream_t stream) const override
    {
        if (!result_owner) { fail("gather produced no result"); }
        auto const result = result_owner->view();
        if (result.type().id() != gamax::type_id::INT64 || result.size() != map_->num_rows()) {
            fail("gather result shape/type mismatch");
        }
        gamax::device_buffer mismatch(sizeof(int), stream, gamax::buffer_init::zero);
        validate_gather_kernel<<<validation_grid(result.size()), 256, 0, stream>>>(
            map_->view()[0].data<gamax::size_type>(), input_->view()[0].data<std::int64_t>(),
            result.data<std::int64_t>(), result.size(), static_cast<int*>(mismatch.data()));
        cuda_check(cudaGetLastError(), "gather validation launch");
        int host_mismatch = 0;
        cuda_check(cudaMemcpyAsync(&host_mismatch, mismatch.data(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   "gather validation copy");
        cuda_check(cudaStreamSynchronize(stream), "gather validation synchronization");
        if (host_mismatch) { fail("gather correctness validation failed"); }
    }

private:
    std::unique_ptr<gamax::table> input_;
    std::unique_ptr<gamax::table> map_;
    gamax::column_ptr output_;
    std::uint64_t size_{};
    std::uint64_t chunk_size_{};
};

class GatherDescriptor final : public OperationDescriptor {
public:
    std::string stable_name() const override { return k_gather_name; }
    std::string axis_name() const override { return "num_inputs=gather_map_size"; }
    std::string config_string(GenericConfig const& config) const override
    {
        return "ValueType=int64_t;algo=thrust;num_inputs=" + std::to_string(config.size()) +
               ";gather_map_size=" + std::to_string(config.size());
    }
    void validate_config(GenericConfig const& config) const override
    {
        if (config.size_exponent < 1 || config.size_exponent > 61) {
            fail("gather size exponent must be in [1, 61]");
        }
    }
    std::size_t estimate_memory(GenericConfig const& config) const override
    {
        return static_cast<std::size_t>(config.size()) * sizeof(std::int64_t) * 4;
    }
    std::unique_ptr<PreparedOperation> prepare(
        GenericConfig const& config, cudaStream_t setup_stream) const override
    {
        auto const size = static_cast<gamax::size_type>(config.size());
        auto input_type = gamax::data_type(gamax::type_id::INT64);
        auto map_type = gamax::data_type(gamax::type_to_id<gamax::size_type>());
        auto input = gamax::datagen::device::pk_uniform_id(
            input_type, size, 1, k_data_seed, setup_stream);
        auto map = gamax::datagen::device::fk_uniform_id(
            map_type, size, size, 1, 1.0, k_data_seed + 1, setup_stream);
        return std::make_unique<PreparedGather>(std::move(input), std::move(map));
    }
};

using BenchmarkMultiset = gamax::containers::multiset<
    std::int64_t, gamax::size_type,
    gamax::rmm_cuco_allocator<gamax::size_type>, 2>;

class PreparedMultisetInsert final : public PreparedOperation {
public:
    PreparedMultisetInsert(std::unique_ptr<gamax::table> build, cudaStream_t setup_stream)
        : build_{std::move(build)}, allocator_{setup_stream},
          set_{build_->view()[0], 0.5, gamax::sentinel<gamax::size_type>(),
               allocator_.get(), setup_stream} {}

    void reset(cudaStream_t stream) override
    {
        set_.clear_async(stream);
        cuda_check(cudaGetLastError(), "multiset clear_async launch");
    }

    gamax::column_ptr enqueue(cudaStream_t stream) override
    {
        set_.insert_with_tids_async(stream);
        cuda_check(cudaGetLastError(), "multiset insert_async launch");
        return {};
    }

    void begin_chunked(std::uint64_t chunk_size, cudaStream_t) override
    {
        size_ = static_cast<std::uint64_t>(build_->num_rows());
        chunk_size_ = effective_chunk_size(size_, chunk_size);
    }

    std::uint64_t chunk_count() const override { return chunk_count_for(size_, chunk_size_); }

    void enqueue_chunk(std::uint64_t chunk_index, cudaStream_t stream) override
    {
        auto const first = chunk_index * chunk_size_;
        auto const last = std::min(size_, first + chunk_size_);
        set_.insert_with_tids_async(static_cast<gamax::size_type>(first),
                                    static_cast<gamax::size_type>(last), stream);
        cuda_check(cudaGetLastError(), "multiset ranged insert_async launch");
    }

    gamax::column_ptr finish_chunked() override { return {}; }

    void validate(gamax::column_ptr const& result, cudaStream_t stream) const override
    {
        if (result) { fail("multiset insert unexpectedly produced a column result"); }
        auto const count = set_.count(build_->view()[0], stream);
        if (count != static_cast<std::size_t>(build_->num_rows())) {
            fail("multiset insert validation failed: expected " +
                 std::to_string(build_->num_rows()) + " matches, found " +
                 std::to_string(count));
        }
    }

private:
    std::unique_ptr<gamax::table> build_;
    gamax::rmm_allocator_handle<gamax::size_type> allocator_;
    mutable BenchmarkMultiset set_;
    std::uint64_t size_{};
    std::uint64_t chunk_size_{};
};

class MultisetInsertDescriptor final : public OperationDescriptor {
public:
    std::string stable_name() const override { return k_multiset_insert_name; }
    std::string axis_name() const override { return "unique_keys"; }
    std::string config_string(GenericConfig const& config) const override
    {
        return "Key=int64_t;CGSize=2;build_dup=1;load_factor=0.5;unique_keys=" +
               std::to_string(config.size()) + ";timed=insert_async";
    }
    void validate_config(GenericConfig const& config) const override
    {
        if (config.size_exponent < 1 || config.size_exponent > 60) {
            fail("multiset insert unique-key exponent must be in [1, 60]");
        }
    }
    std::size_t estimate_memory(GenericConfig const& config) const override
    {
        return static_cast<std::size_t>(config.size()) * 32;
    }
    std::unique_ptr<PreparedOperation> prepare(
        GenericConfig const& config, cudaStream_t setup_stream) const override
    {
        auto const unique_keys = static_cast<gamax::size_type>(config.size());
        auto key_type = gamax::data_type(gamax::type_id::INT64);
        auto build = gamax::datagen::device::fk_uniform_id(
            key_type, unique_keys, unique_keys, 1, 1.0, k_data_seed, setup_stream);
        return std::make_unique<PreparedMultisetInsert>(std::move(build), setup_stream);
    }
};

class PreparedMultisetRetrieve final : public PreparedOperation {
public:
    PreparedMultisetRetrieve(
        std::unique_ptr<gamax::table> build,
        std::unique_ptr<gamax::table> probe,
        cudaStream_t setup_stream)
        : build_{std::move(build)}, probe_{std::move(probe)},
          output_{gamax::make_column<gamax::size_type>(
              probe_->num_rows(), setup_stream, gamax::buffer_init::uninitialized)},
          allocator_{setup_stream},
          set_{build_->view()[0], 0.5, gamax::sentinel<gamax::size_type>(),
               allocator_.get(), setup_stream} {
        set_.insert_with_tids(setup_stream);
    }

    gamax::column_ptr enqueue(cudaStream_t stream) override
    {
        set_.find_async(probe_->view()[0], output_->mutable_view(), stream);
        cuda_check(cudaGetLastError(), "multiset find_async launch");
        return {};
    }

    void begin_chunked(std::uint64_t chunk_size, cudaStream_t) override
    {
        size_ = static_cast<std::uint64_t>(probe_->num_rows());
        chunk_size_ = effective_chunk_size(size_, chunk_size);
    }

    std::uint64_t chunk_count() const override { return chunk_count_for(size_, chunk_size_); }

    void enqueue_chunk(std::uint64_t chunk_index, cudaStream_t stream) override
    {
        auto const offset = chunk_index * chunk_size_;
        auto const length = std::min(chunk_size_, size_ - offset);
        auto const full_probe = probe_->view()[0];
        gamax::column_view probe_slice(
            full_probe.data<std::int64_t>() + offset,
            static_cast<gamax::size_type>(length), full_probe.type());
        set_.find_async(probe_slice,
                        output_->mutable_view(static_cast<gamax::size_type>(offset)), stream);
        cuda_check(cudaGetLastError(), "multiset chunked find_async launch");
    }

    gamax::column_ptr finish_chunked() override { return {}; }

    void validate(gamax::column_ptr const& result, cudaStream_t stream) const override
    {
        if (result) { fail("multiset retrieve unexpectedly produced an owned result"); }
        gamax::device_buffer mismatch(sizeof(int), stream, gamax::buffer_init::zero);
        validate_multiset_find_kernel<<<validation_grid(probe_->num_rows()), 256, 0, stream>>>(
            build_->view()[0].data<std::int64_t>(), probe_->view()[0].data<std::int64_t>(),
            output_->view().data<gamax::size_type>(), build_->num_rows(), probe_->num_rows(),
            gamax::sentinel<gamax::size_type>(), static_cast<int*>(mismatch.data()));
        cuda_check(cudaGetLastError(), "multiset retrieve validation launch");
        int host_mismatch = 0;
        cuda_check(cudaMemcpyAsync(&host_mismatch, mismatch.data(), sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   "multiset retrieve validation copy");
        cuda_check(cudaStreamSynchronize(stream),
                   "multiset retrieve validation synchronization");
        if (host_mismatch) { fail("multiset retrieve correctness validation failed"); }
    }

private:
    std::unique_ptr<gamax::table> build_;
    std::unique_ptr<gamax::table> probe_;
    gamax::column_ptr output_;
    gamax::rmm_allocator_handle<gamax::size_type> allocator_;
    BenchmarkMultiset set_;
    std::uint64_t size_{};
    std::uint64_t chunk_size_{};
};

class MultisetRetrieveDescriptor final : public OperationDescriptor {
public:
    std::string stable_name() const override { return k_multiset_retrieve_name; }
    std::string axis_name() const override { return "probe_keys"; }
    std::string config_string(GenericConfig const& config) const override
    {
        return "Key=int64_t;CGSize=2;build_dup=1;load_factor=0.5;hit_ratio=1.0;probe_keys=" +
               std::to_string(config.size()) + ";unique_keys=" +
               std::to_string(config.size() / 4) + ";timed=find_async";
    }
    void validate_config(GenericConfig const& config) const override
    {
        if (config.size_exponent < 2 || config.size_exponent > 60) {
            fail("multiset retrieve probe-key exponent must be in [2, 60]");
        }
    }
    std::size_t estimate_memory(GenericConfig const& config) const override
    {
        return static_cast<std::size_t>(config.size()) * 32;
    }
    std::unique_ptr<PreparedOperation> prepare(
        GenericConfig const& config, cudaStream_t setup_stream) const override
    {
        auto const probe_keys = static_cast<gamax::size_type>(config.size());
        auto const unique_keys = probe_keys / 4;
        auto key_type = gamax::data_type(gamax::type_id::INT64);
        auto build = gamax::datagen::device::fk_uniform_id(
            key_type, unique_keys, unique_keys, 1, 1.0, k_data_seed, setup_stream);
        auto probe = gamax::datagen::device::fk_uniform_id(
            key_type, unique_keys, probe_keys, 1, 1.0, 39, setup_stream);
        return std::make_unique<PreparedMultisetRetrieve>(
            std::move(build), std::move(probe), setup_stream);
    }
};

} // namespace

OperationRegistry::OperationRegistry()
{
    add(std::make_unique<ExprDescriptor>());
    add(std::make_unique<GatherDescriptor>());
    add(std::make_unique<MultisetInsertDescriptor>());
    add(std::make_unique<MultisetRetrieveDescriptor>());
}

OperationDescriptor const& OperationRegistry::lookup(std::string const& name) const
{
    auto const it = operations_.find(name);
    if (it == operations_.end()) { fail("unknown operation: " + name); }
    return *it->second;
}

void OperationRegistry::list(std::ostream& out) const
{
    for (auto const& [name, _] : operations_) { out << name << '\n'; }
}

void OperationRegistry::add(std::unique_ptr<OperationDescriptor> descriptor)
{
    auto name = descriptor->stable_name();
    operations_.emplace(std::move(name), std::move(descriptor));
}

} // namespace gamax::benchmark::concurrency
