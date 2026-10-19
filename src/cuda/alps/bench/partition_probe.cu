// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_partition_probe — break down the cost of the shuffle's "partition" stage.
//
// The distributed shuffle's partition_ms covers: make-dest (hash key -> rank),
// arg_partition (CUB radix argsort of (dest,index) pairs), one gather per column
// (random-access reorder), and compute_partition_offsets (histogram + scan). This
// probe times each sub-step in isolation (single GPU, no MPI transfer) so we can
// see what actually dominates -- partitioning into `world` buckets is only a few
// bits, yet the stage runs a full sort + per-column gathers.
//
//   ./alps_partition_probe --exp-min 22 --exp-max 25 --world 4 --ncols 2 --iters 20
//
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "memory/column.hpp"
#include "operators/partition/partition.hpp"
#include "utils/transform.hpp"

#include "alps/context.hpp"
#include "alps/error.hpp"

namespace {
using gamax::size_type;

// dest[i] = bit-mixed hash(i) reduced into [0, world): ~uniform, like a real
// key hash, so the post-sort permutation is scattered (not strided).
__global__ void fill_dest(int32_t* d, size_type n, int world) {
    size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t x = static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ull;
    x ^= x >> 29; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 32;
    d[i] = static_cast<int32_t>(x % static_cast<uint64_t>(world));
}

float ms_between(cudaEvent_t a, cudaEvent_t b) { float m = 0; cudaEventElapsedTime(&m, a, b); return m; }
long long arg_ll(const char* v) { return std::strtoll(v, nullptr, 10); }
double mean_of(const std::vector<double>& v) { double s=0; for(double x:v) s+=x; return v.empty()?0:s/v.size(); }
} // namespace

int main(int argc, char** argv) {
    int exp_min = 22, exp_max = 25, world = 4, ncols = 2, iters = 20, warmup = 5;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]{ if (i+1>=argc) throw std::runtime_error("missing val "+k); return argv[++i]; };
        if      (k=="--exp-min") exp_min=(int)arg_ll(nx());
        else if (k=="--exp-max") exp_max=(int)arg_ll(nx());
        else if (k=="--world")   world  =(int)arg_ll(nx());
        else if (k=="--ncols")   ncols  =(int)arg_ll(nx());
        else if (k=="--iters")   iters  =(int)arg_ll(nx());
        else if (k=="--warmup")  warmup =(int)arg_ll(nx());
    }

    alps::configure_process();
    MPI_Init(&argc, &argv);
    alps::Context ctx = alps::make_context();
    cudaStream_t stream = ctx.stream;

    int rank_bits = 0; while ((1 << rank_bits) < world) ++rank_bits;
    if (ctx.rank == 0)
        std::printf("# world=%d rank_bits=%d ncols=%d iters=%d  (ms, single-GPU, mean over iters)\n",
                    world, rank_bits, ncols, iters);

    cudaEvent_t e0{}, e1{}, e2{}, e3{}, e4{};
    for (cudaEvent_t* e : {&e0,&e1,&e2,&e3,&e4}) ALPS_CUDACHECK(cudaEventCreate(e));

    for (int e = exp_min; e <= exp_max; ++e) {
        const size_type n = size_type{1} << e;

        auto dest = gamax::make_column<int32_t>(n, stream);
        { int t=256; long long b=(n+t-1)/t; fill_dest<<<(unsigned)b,t,0,stream>>>(dest->data<int32_t>(), n, world); }
        std::vector<std::unique_ptr<gamax::column>> cols(ncols);
        for (int c = 0; c < ncols; ++c) cols[c] = gamax::make_column<int32_t>(n, stream);
        ALPS_CUDACHECK(cudaStreamSynchronize(stream));

        for (int w = 0; w < warmup; ++w) {
            auto p = gamax::partition::arg_partition(
                dest->view(), 0, rank_bits, stream, nullptr, 0);
            for (int c = 0; c < ncols; ++c) auto g = gamax::util::device::gather(p.second->view(), cols[c]->view(), stream);
            auto off = gamax::partition::compute_partition_offsets(p.first->view(), 0, rank_bits, stream);
            ALPS_CUDACHECK(cudaStreamSynchronize(stream));
        }

        std::vector<double> argpart, gather, offsets;
        for (int it = 0; it < iters; ++it) {
            ALPS_CUDACHECK(cudaEventRecord(e0, stream));
            auto p = gamax::partition::arg_partition(
                dest->view(), 0, rank_bits, stream, nullptr, 0);
            ALPS_CUDACHECK(cudaEventRecord(e1, stream));
            for (int c = 0; c < ncols; ++c) auto g = gamax::util::device::gather(p.second->view(), cols[c]->view(), stream);
            ALPS_CUDACHECK(cudaEventRecord(e2, stream));
            auto off = gamax::partition::compute_partition_offsets(p.first->view(), 0, rank_bits, stream);
            ALPS_CUDACHECK(cudaEventRecord(e3, stream));
            ALPS_CUDACHECK(cudaEventSynchronize(e3));
            argpart.push_back(ms_between(e0, e1));
            gather.push_back(ms_between(e1, e2));
            offsets.push_back(ms_between(e2, e3));
        }

        const double a = mean_of(argpart), g = mean_of(gather), o = mean_of(offsets);
        std::printf("  n=2^%-2d (%9lld)  arg_partition %7.3f  gather(x%d) %7.3f  offsets %7.3f  | sum %7.3f ms\n",
                    e, (long long)n, a, ncols, g, o, a + g + o);
    }

    for (cudaEvent_t ev : {e0,e1,e2,e3,e4}) cudaEventDestroy(ev);
    alps::destroy_context(ctx);
    MPI_Finalize();
    return 0;
}
