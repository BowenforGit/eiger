// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "alps/collectives/ce_pull_options.hpp"
#include "alps/collectives/table_exchange.hpp"
#include "alps/containers/fabric_peer_buffer.hpp"
#include "alps/context.hpp"
#include "memory/column.hpp"

namespace alps::collectives {

struct CePullStats {
    std::size_t slot_capacity_bytes {0};
    std::size_t arena_capacity_bytes {0};
    std::size_t control_capacity_bytes {0};
    std::size_t peak_requested_slot_bytes {0};
    std::uint64_t arena_growth_count {0};
    std::uint64_t arena_overflow_count {0};
    std::uint64_t enqueued_columns {0};
    std::uint64_t remote_bytes {0};
    std::uint64_t slot_reuse_count {0};
    double arena_growth_ms {0.0};
    bool growth_frozen {false};
};

class CePullSessionImpl;

// Persistent per-Context state shared by all left/right shuffle sessions.
// Destruction is best-effort local cleanup; orderly users must call free()
// collectively before MPI_Finalize (destroy_context does this).
class CePullExchangeContext {
public:
    CePullExchangeContext() = default;
    ~CePullExchangeContext();

    CePullExchangeContext(const CePullExchangeContext&) = delete;
    CePullExchangeContext& operator=(const CePullExchangeContext&) = delete;

    void freeze_growth() noexcept;
    void allow_growth() noexcept;
    CePullStats stats() const noexcept;
    cudaStream_t transfer_stream() const noexcept { return transfer_stream_; }
    void free(const Context& ctx);

private:
    struct Lease {
        int slot {-1};
        std::size_t offset {0};
        std::size_t capacity {0};
    };

    struct Slot {
        cudaEvent_t free {nullptr};
        cudaEvent_t ready_published {nullptr};
        cudaEvent_t consumed_published {nullptr};
    };

    void initialize(const Context& ctx, const CePullOptions& options);
    void ensure_slot_count(std::size_t count, const Context& ctx);
    void ensure_slot_capacity(
        std::size_t bytes, std::size_t slot_count, const Context& ctx);
    void ensure_arena_capacity(std::size_t bytes, const Context& ctx);
    void wait_control_words(
        cudaStream_t stream, std::size_t first_index, int count,
        unsigned long long epoch);
    void begin_full_staging_batch(
        std::size_t operations, const Context& ctx,
        cudaStream_t preparation_stream, const CePullOptions& options);
    std::vector<Lease> reserve_full_staging_leases(
        const std::vector<std::size_t>& local_bytes, const Context& ctx,
        cudaStream_t preparation_stream);
    void seal_full_staging_batch();
    void finish_full_staging_batch();
    void cancel_full_staging_batch() noexcept;
    void teardown_local() noexcept;

    containers::FabricPeerBuffer arena_;
    containers::FabricPeerBuffer control_;
    std::vector<Slot> slots_;
    cudaStream_t transfer_stream_ {nullptr};
    cudaStream_t reclaim_stream_ {nullptr};
    cudaEvent_t full_staging_batch_free_ {nullptr};
    CePullOptions options_ {};
    CePullStats stats_ {};
    std::size_t bounded_slot_capacity_bytes_ {0};
    std::size_t bounded_slot_count_ {0};
    std::uint64_t next_operation_ {0};
    std::size_t full_staging_expected_operations_ {0};
    std::size_t full_staging_reserved_operations_ {0};
    std::size_t full_staging_reserved_bytes_ {0};
    int world_ {0};
    int rank_ {0};
    bool initialized_ {false};
    bool full_staging_batch_active_ {false};

    friend class CePullSessionImpl;
    friend class CePullShuffleSession;
    friend void begin_ce_pull_full_staging_batch(
        const Context&, std::size_t, cudaStream_t, const CePullOptions&);
    friend void seal_ce_pull_full_staging_batch(const Context&);
    friend void finish_ce_pull_full_staging_batch(const Context&);
    friend void cancel_ce_pull_full_staging_batch(const Context&) noexcept;
};

// A host-planned, allocation-free-at-enqueue shuffle of an ordered set of
// columns sharing one HashPartitionPlan. The plan and input column storage must
// outlive the session's enqueued preparation work.
class CePullShuffleSession {
public:
    ~CePullShuffleSession();
    CePullShuffleSession(CePullShuffleSession&&) noexcept;
    CePullShuffleSession& operator=(CePullShuffleSession&&) noexcept;

    CePullShuffleSession(const CePullShuffleSession&) = delete;
    CePullShuffleSession& operator=(const CePullShuffleSession&) = delete;

    std::size_t num_columns() const noexcept;
    void prepare_column(std::size_t ordinal);
    void enqueue_pull(std::size_t ordinal);
    void enqueue_column(std::size_t ordinal);
    gamax::column& output(std::size_t ordinal);
    const gamax::column& output(std::size_t ordinal) const;
    cudaEvent_t ready_event(std::size_t ordinal) const;
    void wait_on(std::size_t ordinal, cudaStream_t consumer_stream) const;
    void synchronize_reclamation() const;
    gamax::column_ptr release_output(
        std::size_t ordinal, cudaStream_t lifetime_stream);

private:
    explicit CePullShuffleSession(std::unique_ptr<CePullSessionImpl> impl);
    std::unique_ptr<CePullSessionImpl> impl_;

    friend std::unique_ptr<CePullShuffleSession> begin_ce_pull_shuffle(
        const HashPartitionPlan&, std::vector<gamax::column_view>,
        const Context&, cudaStream_t, const CePullOptions&);
    friend std::unique_ptr<gamax::table> ce_pull_exchange_partitioned(
        gamax::table_view, CommPattern, const Context&, cudaStream_t,
        const CePullOptions&);
    friend std::unique_ptr<gamax::table> ce_pull_allgather(
        gamax::table_view, const Context&, cudaStream_t,
        const CePullOptions&, float*);
};

// Collective host-planning boundary. Completes the receive pattern, exchanges
// compact sender-source offsets, allocates outputs/descriptors, and grows the
// persistent arena if permitted. Rank-local planning/allocation errors are
// reduced before a later collective is entered. enqueue_column performs none
// of those tasks.
std::unique_ptr<CePullShuffleSession> begin_ce_pull_shuffle(
    const HashPartitionPlan& plan,
    std::vector<gamax::column_view> ordered_columns,
    const Context& ctx,
    cudaStream_t preparation_stream,
    const CePullOptions& options = {});

// Blocking/bounded adapters used by the global CE_PULL backend. They enqueue
// all columns through the shared two-slot ring and make the caller stream wait
// for every output before returning, so downstream Legacy compute cannot
// overlap the collective.
std::unique_ptr<gamax::table> ce_pull_exchange_partitioned(
    gamax::table_view partitioned,
    CommPattern pattern,
    const Context& ctx,
    cudaStream_t stream,
    const CePullOptions& options = {});

std::unique_ptr<gamax::table> ce_pull_allgather(
    gamax::table_view input,
    const Context& ctx,
    cudaStream_t stream,
    const CePullOptions& options = {},
    float* comm_ms = nullptr);

// A full-staging batch is a host-planned group of sessions whose columns all
// receive distinct VMM leases. Calls are collective and ordered consistently
// across ranks. begin/reset and seal happen before device preparation;
// finish is called after every pull has been submitted. In full-staging mode
// it publishes one whole-shuffle completion doorbell per receiver and records
// the arena-reuse edge after every sender observes all receivers.
void begin_ce_pull_full_staging_batch(
    const Context& ctx,
    std::size_t operations,
    cudaStream_t preparation_stream,
    const CePullOptions& options);
void seal_ce_pull_full_staging_batch(const Context& ctx);
void finish_ce_pull_full_staging_batch(const Context& ctx);
void cancel_ce_pull_full_staging_batch(const Context& ctx) noexcept;

void freeze_ce_pull_arena(const Context& ctx);
void allow_ce_pull_arena_growth(const Context& ctx);
CePullStats ce_pull_stats(const Context& ctx);
cudaStream_t ce_pull_transfer_stream(const Context& ctx);

} // namespace alps::collectives
