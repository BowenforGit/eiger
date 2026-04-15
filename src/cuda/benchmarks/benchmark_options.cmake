# ILP benchmark: bypass L1 cache for all loads/stores by default
if(USE_NVIDIA)
    target_compile_options(benchmark_ilp PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=cg,-dscm=wt>)
endif()
