// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

// alps_join_probe — single-GPU repro of the local-join failure at large output.
//
// Builds a small unique PK and a large FK (card = 2^fk-exp, all matching), then
// runs gamax::join::inner_eq_join. The distributed join's per-rank local join
// hits cudaErrorIllegalAddress once the output reaches ~2^30 rows; this isolates
// it to a single GPU (no MPI/shuffle) so it can be debugged / sanitized.
//
//   GAMAX_RMM_RESOURCE=arena GAMAX_MEM_RESERVE_BYTES=8000000000 \
//     ./alps_join_probe --pk-exp 24 --fk-exp 30 --algo hash
//
#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "memory/table.hpp"
#include "operators/join/join.hpp"
#include "operators/join/inner_join.hpp"
#include "types/data_type.hpp"
#include "data_gen/table_gen.hpp"

#include "alps/context.hpp"

int main(int argc, char** argv) {
    int pk_exp = 24, fk_exp = 30;
    std::string algo = "hash";
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() -> const char* { return argv[++i]; };
        if      (k == "--pk-exp") pk_exp = std::atoi(nx());
        else if (k == "--fk-exp") fk_exp = std::atoi(nx());
        else if (k == "--algo")   algo   = nx();
    }

    alps::configure_process();          // allocator from env (no MPI needed here)
    cudaSetDevice(0);
    gamax::prewarm_device_memory_resource();

    const gamax::size_type M = (gamax::size_type{1} << pk_exp);
    const gamax::size_type N = (gamax::size_type{1} << fk_exp);
    const gamax::data_type i32{gamax::type_id::INT32};

    auto pk = gamax::datagen::device::pk_uniform_id(i32, M, 2, 1, cudaStreamDefault, 0);
    auto fk = gamax::datagen::device::fk_uniform_id(
        i32, M, N, 2, 1.0, 2, cudaStreamDefault, 0);
    cudaDeviceSynchronize();

    gamax::join::join_hint hint;
    hint.join_algo   = (algo == "smj") ? gamax::join::SORT : gamax::join::HASH;
    hint.mat_algo    = gamax::join::GFUR;
    hint.unique_left = true;
    hint.max_out     = 0;

    std::printf("join: pk=2^%d (%lld)  fk=2^%d (%lld)  algo=%s  mat=gfur\n",
                pk_exp, (long long)M, fk_exp, (long long)N, algo.c_str());
    std::fflush(stdout);

    auto out = gamax::join::inner_eq_join(
        pk->view(), fk->view(), hint, cudaStreamDefault);
    cudaError_t e = cudaDeviceSynchronize();
    std::printf("out rows = %lld  (expect %lld)  cuda=%s\n",
                (long long)out->num_rows(), (long long)N, cudaGetErrorString(e));
    return e == cudaSuccess ? 0 : 1;
}
