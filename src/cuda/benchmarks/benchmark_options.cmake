# Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE and THIRD_PARTY_NOTICES.md for license information.

# ILP benchmark: bypass L1 cache for all loads/stores by default.
target_compile_options(benchmark_ilp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=cg,-dscm=wt>)
